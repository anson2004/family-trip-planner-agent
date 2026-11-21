"""The tools enforce the trip's own limits, the search quota and the flight/stay match, whatever the LLM asks for."""

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import respx

from tests.test_core import DATES, load, make_request
from travel_plan_agent.cache import JsonCache
from travel_plan_agent.models import PackagePick, PlanDraft, TripRequest, Usage
from travel_plan_agent.origins import resolve_origin
from travel_plan_agent.providers.serpapi import SERPAPI_URL, SerpApiClient
from travel_plan_agent.session import TripSession
from travel_plan_agent.tools import QUOTA_USED_UP, build_tools

FLIGHTS: dict[str, Any] = {"departure_airports": ["TMP", "HEL"], "arrival_airports": ["LPA"]}


def _tools(tmp_path: Path, request: TripRequest, max_live_calls: int | None = None) -> tuple[TripSession, Any]:
    session = TripSession(request=request, origin=resolve_origin("Tampere"))
    client = SerpApiClient("key", JsonCache(tmp_path, 1), max_live_calls=max_live_calls)
    return session, {t.name: t for t in build_tools(session, client, None)}


def _mock_serpapi(calendar: dict[str, Any] | None = None) -> respx.Route:
    def respond(request: httpx.Request) -> httpx.Response:
        engine = request.url.params["engine"]
        if engine == "google_flights_calendar":
            return httpx.Response(200, json=calendar or load("calendar.json"))
        return httpx.Response(200, json=load("flights.json" if engine == "google_flights" else "hotels.json"))

    return respx.get(SERPAPI_URL).mock(side_effect=respond)


@respx.mock
def test_trip_limits_apply_even_when_the_llm_asks_for_less(tmp_path: Path) -> None:
    route = _mock_serpapi()
    _, tools = _tools(tmp_path, make_request(max_stops=0, hotel_min_rating=4.5))
    tools["search_flights"].invoke({**FLIGHTS, "max_stops": 2})
    tools["search_hotels"].invoke({"location": "Maspalomas", "min_rating": 3.5})
    flights, hotels = (c.request.url.params for c in route.calls)
    assert flights["stops"] == "1"  # nonstop only
    assert hotels["rating"] == "9"  # 4.5+


@respx.mock
def test_quota_still_allows_cached_searches(tmp_path: Path) -> None:
    route = _mock_serpapi()
    _, tools = _tools(tmp_path, make_request(), max_live_calls=1)
    first = tools["search_flights"].invoke(FLIGHTS)
    assert "(searches left: 0)" in first
    assert tools["search_flights"].invoke(FLIGHTS).startswith("2 flights")  # same search, from the cache
    assert tools["search_hotels"].invoke({"location": "Maspalomas"}) == QUOTA_USED_UP
    assert route.call_count == 1


@respx.mock
def test_hotels_are_priced_only_with_flights_landing_nearby(tmp_path: Path) -> None:
    _mock_serpapi()
    session, tools = _tools(tmp_path, make_request(destinations=["Gran Canaria", "Tenerife"]))
    tools["search_flights"].invoke(FLIGHTS)  # both flights land at LPA

    unknown = tools["search_hotels"].invoke({"location": "Maspalomas"})
    assert "with F" not in unknown and "Pass near_airports" in unknown
    tenerife = tools["search_hotels"].invoke({"location": "Adeje", "near_airports": ["tfs"]})
    assert "with F" not in tenerife and "Pass near_airports" not in tenerife
    gran_canaria = tools["search_hotels"].invoke({"location": "Maspalomas", "near_airports": ["LPA"]})
    assert "with F1 (TMP): 2500 total" in gran_canaria
    assert "2 stays in Maspalomas" in gran_canaria and "first 2 shown" in gran_canaria

    # H3 is the first Tenerife stay, H5 the first Gran Canaria one: a package with an LPA flight is rejected.
    draft = PlanDraft(
        summary="s",
        origin_note="o",
        packages=[
            PackagePick(flight_id="F1", hotel_id="H3", why="x"),
            PackagePick(flight_id="F1", hotel_id="H5", why="y"),
        ],
    )
    plan = session.build_plan(draft, Usage(llm_model="t"))
    ids = [p.hotel.id for p in plan.packages]
    assert "H3" not in ids and "H5" in ids  # the top-up also only uses stays near LPA
    assert "lands at LPA, but the stay is near TFS" in plan.warnings[0]


@respx.mock
def test_cheapest_dates_only_lists_trips_in_the_window(tmp_path: Path) -> None:
    calendar = {
        "prices": [
            {"outbound_date": "2027-02-17", "return_date": "2027-02-24", "price": 1180},
            {"outbound_date": "2027-02-25", "return_date": "2027-03-04", "price": 900},  # leaves too late
            {"outbound_date": "2027-02-15", "return_date": "2027-02-20", "price": 800},  # wrong length
        ]
    }
    _mock_serpapi(calendar)
    request = make_request(depart=date(2027, 2, 14), return_date=date(2027, 2, 28), trip_days=7)
    _, tools = _tools(tmp_path, request)
    out = tools["find_cheapest_dates"].invoke({"departure_airport": "TMP", "arrival_airport": "LPA"})
    assert "2027-02-17 -> 2027-02-24" in out and "02-25" not in out and "02-15" not in out
    assert out.startswith("Cheapest 7-night trips")


def test_fixed_dates_unchanged() -> None:
    assert make_request().dates_from(make_request().depart) == DATES
