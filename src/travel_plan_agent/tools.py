"""LangChain tools the agent can call. The party and the travel window are fixed by the session, not by the LLM."""

from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, tool

from travel_plan_agent.models import FlightOption, HotelOption
from travel_plan_agent.providers.serpapi import SerpApiClient, SerpApiError
from travel_plan_agent.providers.tavily import TavilyClient
from travel_plan_agent.session import TripSession

MAX_LISTED = 8


def _duration(minutes: int | None) -> str:
    return "?" if minutes is None else f"{minutes // 60}h{minutes % 60:02d}"


def _flight_line(opt: FlightOption, transfer: float) -> str:
    via = ", ".join(leg.to_airport for leg in opt.outbound_legs[:-1])
    stops = "nonstop" if opt.stops == 0 else f"{opt.stops} stop(s) via {via}"
    first, last = opt.outbound_legs[0], opt.outbound_legs[-1]
    line = (
        f"{opt.id} | {opt.price_total:.0f} {opt.currency} round trip, whole party | "
        f"{opt.departure_airport}->{opt.arrival_airport} | {stops} | {_duration(opt.total_duration_min)} | "
        f"{', '.join(opt.airlines)} | outbound {first.departs} -> {last.arrives}"
    )
    if transfer:
        line += f" | +{transfer:.0f} {opt.currency} ground transfer to {opt.departure_airport}"
    return line


def _hotel_line(opt: HotelOption) -> str:
    parts = [
        f"{opt.id} | {opt.name} ({opt.kind})",
        f"{opt.total_price:.0f} {opt.currency} total stay",
    ]
    if opt.price_per_night is not None:
        parts.append(f"{opt.price_per_night:.0f}/night")
    if opt.rating is not None:
        parts.append(f"rating {opt.rating} ({opt.reviews or 0} reviews)")
    if opt.stars:
        parts.append(f"{opt.stars}-star")
    if opt.amenities:
        parts.append("amenities: " + ", ".join(opt.amenities[:5]))
    if opt.sellers:
        parts.append("sold by: " + ", ".join(opt.sellers[:3]))
    return " | ".join(parts)


def build_tools(
    session: TripSession, serpapi: SerpApiClient, tavily: TavilyClient | None, max_calls: int
) -> list[BaseTool]:
    request = session.request

    def quota_left() -> int:
        return max_calls - serpapi.live_calls

    def trip_dates(depart_date: str | None) -> tuple[date, date] | str:
        """The travel dates to search, or an error message for the agent."""
        if not request.flexible:
            return request.dates_from(request.depart)
        if not depart_date:
            return (
                f"depart_date is required: pick one between {request.depart} and {request.latest_departure} "
                "(use find_cheapest_dates first)."
            )
        try:
            return request.dates_from(date.fromisoformat(depart_date))
        except ValueError as exc:
            return f"Invalid depart_date: {exc}"

    @tool(parse_docstring=True)
    def find_cheapest_dates(departure_airport: str, arrival_airport: str) -> str:
        """Show which departure dates in the travel window are cheapest for one route. Costs one search.

        Args:
            departure_airport: One IATA code to fly from, e.g. "TMP".
            arrival_airport: One IATA code to fly to, e.g. "LPA".
        """
        if quota_left() <= 0:
            return "Search quota for this plan is used up. Work with the results you already have."
        session.log(f"Cheapest dates {departure_airport} -> {arrival_airport}")
        try:
            rows, _ = serpapi.search_calendar(request, departure_airport.upper(), arrival_airport.upper())
        except SerpApiError as exc:
            return f"Date search failed: {exc}"
        if not rows:
            return "No prices found for this route. Try another airport pair."
        lines = [
            f"{r.outbound_date} -> {r.return_date}: ~{r.price:.0f} {request.currency}"
            + (" (low)" if r.low_price else "")
            for r in rows[:8]
        ]
        return (
            f"Cheapest {request.nights}-day trips {departure_airport}->{arrival_airport}, approximate prices "
            f"(searches left: {quota_left()}):\n" + "\n".join(lines)
        )

    @tool(parse_docstring=True)
    def search_flights(
        departure_airports: list[str],
        arrival_airports: list[str],
        max_stops: int | None = None,
        depart_date: str | None = None,
    ) -> str:
        """Search round-trip flights for the party. Prices are totals for everyone.

        Args:
            departure_airports: IATA codes to fly from, e.g. ["TMP", "HEL"]. Several codes are searched in one call.
            arrival_airports: IATA codes near the destination, e.g. ["LPA", "TFS"]. Up to 4 codes per call.
            max_stops: 0 for nonstop only, 1 or 2 to limit stops, null for any.
            depart_date: Outbound date YYYY-MM-DD when the dates are flexible (return follows from trip length).
        """
        dates = trip_dates(depart_date)
        if isinstance(dates, str):
            return dates
        if quota_left() <= 0:
            return "Search quota for this plan is used up. Work with the results you already have."
        session.log(f"Flights {','.join(departure_airports)} -> {','.join(arrival_airports)} ({dates[0]})")
        try:
            found, _ = serpapi.search_flights(
                request, dates, departure_airports, arrival_airports[:4], max_stops, id_prefix="F"
            )
        except SerpApiError as exc:
            return f"Flight search failed: {exc}"
        if not found:
            return "No flights found. Try other airports, other dates or allow more stops."
        added = session.add_flights(found)
        cheapest = sorted(added, key=lambda f: f.price_total + session.ground_transfer(f))[:MAX_LISTED]
        lines = [_flight_line(f, session.ground_transfer(f)) for f in cheapest]
        return (
            f"{len(added)} flights {dates[0]} -> {dates[1]}, cheapest shown (searches left: {quota_left()}):\n"
            + "\n".join(lines)
        )

    @tool(parse_docstring=True)
    def search_hotels(
        location: str,
        max_price_per_night: int | None = None,
        min_rating: float | None = None,
        sort_by: Literal["price", "rating", "reviews"] = "price",
        depart_date: str | None = None,
    ) -> str:
        """Search hotels and holiday apartments for the party. Prices are for the whole stay.

        Args:
            location: Specific area to stay, e.g. "Playa del Ingles, Gran Canaria" or "family hotel Alanya".
            max_price_per_night: Optional limit per night, in the trip currency.
            min_rating: Optional minimum guest rating: 3.5, 4.0 or 4.5.
            sort_by: "price" for cheapest first, "rating" for best rated first, or "reviews" for most reviewed.
            depart_date: Check-in date YYYY-MM-DD when the dates are flexible. Use the same date as the flights.
        """
        dates = trip_dates(depart_date)
        if isinstance(dates, str):
            return dates
        if quota_left() <= 0:
            return "Search quota for this plan is used up. Work with the results you already have."
        session.log(f"Hotels in {location} ({dates[0]})")
        try:
            found, _ = serpapi.search_hotels(
                request, dates, location, max_price_per_night, min_rating, sort_by, id_prefix="H"
            )
        except SerpApiError as exc:
            return f"Hotel search failed: {exc}"
        if not found:
            return "No hotels found. Try a broader location or a higher price limit."
        added = session.add_hotels(found)[:MAX_LISTED]
        # Price each hotel against the cheapest flight on the same dates from each departure airport,
        # so the agent can see what fits the budget without extra LLM round-trips.
        cheapest: dict[str, FlightOption] = {}
        for f in session.flights.values():
            if f.outbound_date != dates[0]:
                continue
            best = cheapest.get(f.departure_airport)
            if best is None or f.price_total < best.price_total:
                cheapest[f.departure_airport] = f
        lines = []
        for h in added:
            combos = [session.price(f, h) for f in cheapest.values()]
            line = _hotel_line(h)
            if combos:
                line += " | " + "; ".join(
                    f"with {p.flight.id} ({p.flight.departure_airport}): {p.total:.0f} total, {p.budget_status.value}"
                    for p in combos
                )
            lines.append(line)
        return (
            f"{len(added)} stays in {location}, {dates[0]} -> {dates[1]} (searches left: {quota_left()}):\n"
            + "\n".join(lines)
        )

    @tool(parse_docstring=True)
    def search_travel_agencies(query: str, finnish_agencies_only: bool = True) -> str:
        """Search the web for package holidays (flight + hotel) from travel agencies, or airline and hotel deals.

        Args:
            query: What to look for, e.g. "Gran Canaria family package holiday from Tampere February".
            finnish_agencies_only: true to search only Finnish tour operators (Tjäreborg, TUI, Apollo...).
        """
        if tavily is None:
            return "Web search is not set up (no TAVILY_API_KEY)."
        session.log(f"Web search: {query}")
        try:
            results = session.add_web(tavily.search(query, finnish_agencies_only, id_prefix="W"))
        except Exception as exc:  # network or API errors should not stop the plan
            return f"Web search failed: {exc}"
        if not results:
            return "No results."
        return "\n\n".join(f"{r.id} | {r.title}\n{r.url}\n{r.snippet}" for r in results)

    tools: list[BaseTool] = [search_flights, search_hotels, search_travel_agencies]
    if request.flexible:
        tools.insert(0, find_cheapest_dates)
    return tools
