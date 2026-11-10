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

    def build_plan(self, draft: PlanDraft, usage: Usage) -> TravelPlan:
        warnings = list(draft.warnings)
        packages: list[PricedPackage] = []
        seen: set[tuple[str, str]] = set()
        for pick in draft.packages:
            flight, hotel = self.flights.get(pick.flight_id), self.hotels.get(pick.hotel_id)
            if flight is None or hotel is None:
                warnings.append(f"Dropped a package with unknown IDs {pick.flight_id}/{pick.hotel_id}.")
                continue
            if (flight.id, hotel.id) in seen:
                continue
            seen.add((flight.id, hotel.id))
            packages.append(self.price(flight, hotel, pick.why))

        # Packages within budget first, then cheaper ones, then over budget. Otherwise keep the agent's order.
        order = {BudgetStatus.WITHIN: 0, BudgetStatus.UNDER: 1, BudgetStatus.OVER: 2}
        packages.sort(key=lambda p: order[p.budget_status])

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
