"""State for one planning run: every option the tools found, kept by ID.

The LLM only ever sees summaries and IDs. Prices for the final plan always come from here.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from travel_plan_agent.models import (
    AgencyDeal,
    BudgetStatus,
    FlightOption,
    HotelOption,
    PlanDraft,
    PricedPackage,
    TravelPlan,
    TripRequest,
    Usage,
    WebResult,
)
from travel_plan_agent.origins import ResolvedOrigin

MIN_PACKAGES = 5  # the agent often returns only 1-2; the system tops up to this many


def _cheapest_first(p: PricedPackage) -> tuple[bool, float]:
    """Cheaper is better: cheapest first, with packages over the budget maximum last."""
    return p.budget_status is BudgetStatus.OVER, p.total


def _mismatch(flight: FlightOption, hotel: HotelOption) -> str | None:
    """Why this flight and stay can't be one package, or None when they fit."""
    if flight.outbound_date != hotel.check_in or flight.return_date != hotel.check_out:
        return f"the flight ({flight.outbound_date}) and the stay ({hotel.check_in}) are on different dates"
    if hotel.near_airports and flight.arrival_airport not in hotel.near_airports:
        return f"the flight lands at {flight.arrival_airport}, but the stay is near {', '.join(hotel.near_airports)}"
    return None


@dataclass
class TripSession:
    request: TripRequest
    origin: ResolvedOrigin
    log: Callable[[str], None] = lambda _msg: None
    flights: dict[str, FlightOption] = field(default_factory=dict)
    hotels: dict[str, HotelOption] = field(default_factory=dict)
    web: dict[str, WebResult] = field(default_factory=dict)

    def add_flights(self, options: list[FlightOption]) -> list[FlightOption]:
        added = []
        for opt in options:
            opt = opt.model_copy(update={"id": f"F{len(self.flights) + 1}"})
            self.flights[opt.id] = opt
            added.append(opt)
        return added

    def add_hotels(self, options: list[HotelOption]) -> list[HotelOption]:
        added = []
        for opt in options:
            opt = opt.model_copy(update={"id": f"H{len(self.hotels) + 1}"})
            self.hotels[opt.id] = opt
            added.append(opt)
        return added

    def add_web(self, results: list[WebResult]) -> list[WebResult]:
        added = []
        for res in results:
            res = res.model_copy(update={"id": f"W{len(self.web) + 1}"})
            self.web[res.id] = res
            added.append(res)
        return added

    def ground_transfer(self, flight: FlightOption) -> float:
        return self.origin.transfer_cost(flight.departure_airport, self.request.travellers)

    def budget_status(self, total: float) -> BudgetStatus:
        if total > self.request.budget_max:
            return BudgetStatus.OVER
        if total < self.request.budget_min:
            return BudgetStatus.UNDER
        return BudgetStatus.WITHIN

    def price(self, flight: FlightOption, hotel: HotelOption, why: str = "") -> PricedPackage:
        transfer = self.ground_transfer(flight)
        total = round(flight.price_total + hotel.total_price + transfer, 2)
        return PricedPackage(
            flight=flight,
            hotel=hotel,
            ground_transfer=transfer,
            total=total,
            budget_status=self.budget_status(total),
            why=why,
        )

    def _known_match(self, flight: FlightOption, hotel: HotelOption) -> bool:
        """The flight and stay surely go together: same dates, and the flight lands near the stay."""
        if _mismatch(flight, hotel):
            return False
        # Without near_airports, a stay can only be matched when there is a single destination.
        return bool(hotel.near_airports) or len(self.request.destinations) == 1

    def build_plan(self, draft: PlanDraft, usage: Usage) -> TravelPlan:
        warnings = list(draft.warnings)
        packages: list[PricedPackage] = []
        seen: set[tuple[str, str]] = set()
        for pick in draft.packages:
            flight, hotel = self.flights.get(pick.flight_id), self.hotels.get(pick.hotel_id)
            if flight is None or hotel is None:
                warnings.append(f"Dropped a package with unknown IDs {pick.flight_id}/{pick.hotel_id}.")
                continue
            problem = _mismatch(flight, hotel)
            if problem:
                warnings.append(f"Dropped {flight.id}+{hotel.id}: {problem}.")
                continue
            if (flight.id, hotel.id) in seen:
                continue
            seen.add((flight.id, hotel.id))
            packages.append(self.price(flight, hotel, pick.why))

        # Top up with the next cheapest valid combinations, so there is always a choice.
        missing = MIN_PACKAGES - len(packages)
        if missing > 0:
            extra = sorted(
                (
                    self.price(f, h, "Added by the system: the next cheapest combination found.")
                    for f in self.flights.values()
                    for h in self.hotels.values()
                    if (f.id, h.id) not in seen and self._known_match(f, h)
                ),
                key=_cheapest_first,
            )
            # Different stays first (the cheapest package for each), so the options aren't all the same hotel.
            used = {p.hotel.id for p in packages}
            new_stays: list[PricedPackage] = []
            same_stays: list[PricedPackage] = []
            for p in extra:
                (same_stays if p.hotel.id in used else new_stays).append(p)
                used.add(p.hotel.id)
            packages += (new_stays + same_stays)[:missing]

        packages.sort(key=_cheapest_first)
        hostels = dict.fromkeys(p.hotel.name for p in packages if p.hotel.kind == "hostel")
        warnings += [
            f"{name} is a hostel: its price may be for one bed, so check it covers all "
            f"{self.request.travellers} travellers."
            for name in hostels
        ]

        deals = [
            AgencyDeal(result=self.web[d.result_id], quoted_price=d.quoted_price, why=d.why)
            for d in draft.agency_deals
            if d.result_id in self.web
        ]
        return TravelPlan(
            request=self.request,
            summary=draft.summary,
            origin_note=draft.origin_note,
            packages=packages,
            agency_deals=deals,
            warnings=warnings,
            usage=usage,
        )
