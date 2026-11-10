"""Flexible dates: TRIP_DEPART/TRIP_RETURN as a window plus TRIP_DAYS as the trip length."""

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage

from tests.test_agent import ScriptedModel, call
from tests.test_core import load, make_request
from travel_plan_agent.agent import plan_trip
from travel_plan_agent.cache import JsonCache
from travel_plan_agent.config import Settings, TripSettings
from travel_plan_agent.models import PackagePick, PlanDraft, Usage
from travel_plan_agent.origins import resolve_origin
from travel_plan_agent.providers.serpapi import SERPAPI_URL, SerpApiClient, parse_calendar, parse_flights, parse_hotels
from travel_plan_agent.providers.tavily import TAVILY_URL
from travel_plan_agent.session import TripSession
from travel_plan_agent.tools import build_tools

WINDOW: dict[str, Any] = dict(depart=date(2027, 2, 14), return_date=date(2027, 2, 28), trip_days=7)


def test_window_and_trip_length() -> None:
    r = make_request(**WINDOW)
    assert r.flexible and r.nights == 7 and r.latest_departure == date(2027, 2, 21)
    assert r.dates_from(date(2027, 2, 17)) == (date(2027, 2, 17), date(2027, 2, 24))
    with pytest.raises(ValueError, match="between 2027-02-14 and 2027-02-21"):
        r.dates_from(date(2027, 2, 22))
    assert r.dates_label() == "7 days, leaving 14 Feb – 21 Feb 2027 (back by 28 Feb)"
    with pytest.raises(ValueError, match="doesn't fit"):
        make_request(depart=date(2027, 2, 14), return_date=date(2027, 2, 18), trip_days=7)
    fixed = make_request()
    assert not fixed.flexible and fixed.nights == 7
    assert not make_request(trip_days=7).flexible  # window exactly as long as the trip


def test_trip_days_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRIP_DAYS", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "TRIP_TO=Gran Canaria\nTRIP_DEPART=2027-02-14\nTRIP_RETURN=2027-02-28\nTRIP_DAYS=7\nTRIP_BUDGET=3000\n"
    )
    r = TripSettings(_env_file=env).to_request()
    assert r.trip_days == 7 and r.flexible


def test_calendar_params_and_parse() -> None:
    client = SerpApiClient("key", JsonCache(Path("/nonexistent"), 1))
    params = client.calendar_params(make_request(**WINDOW, children_ages=[6, 1]), "TMP", "LPA")
    assert params["engine"] == "google_flights_calendar" and params["round_trip_days"] == 7
    assert (params["date_range_start"], params["date_range_end"]) == ("2027-02-14", "2027-02-21")
    assert params["children"] == 1 and params["infants_on_lap"] == 1
    rows = parse_calendar(load("calendar.json"))
    assert [r.outbound_date.day for r in rows] == [17, 20, 14]  # cheapest first; no-price row skipped
    assert rows[0].low_price and rows[0].return_date == date(2027, 2, 24)


def test_tools_need_a_date_when_flexible(tmp_path: Path) -> None:
    session = TripSession(request=make_request(**WINDOW), origin=resolve_origin("Tampere"))
    client = SerpApiClient("key", JsonCache(tmp_path, 1))
    tools = {t.name: t for t in build_tools(session, client, None, max_calls=5)}
    assert "find_cheapest_dates" in tools
    out = tools["search_flights"].invoke({"departure_airports": ["TMP"], "arrival_airports": ["LPA"]})
    assert "depart_date is required" in out and client.live_calls == 0
    out = tools["search_hotels"].invoke({"location": "Maspalomas", "depart_date": "2027-02-25"})
    assert "Invalid depart_date" in out
    fixed = build_tools(TripSession(request=make_request(), origin=resolve_origin("Tampere")), client, None, 5)
    assert "find_cheapest_dates" not in {t.name for t in fixed}


def test_mismatched_dates_are_dropped() -> None:
    session = TripSession(request=make_request(**WINDOW), origin=resolve_origin("Tampere"))
    session.add_flights(parse_flights(load("flights.json"), "EUR", "x", (date(2027, 2, 17), date(2027, 2, 24))))
    session.add_hotels(parse_hotels(load("hotels.json"), "GC", "EUR", "x", (date(2027, 2, 14), date(2027, 2, 21))))
    draft = PlanDraft(summary="s", origin_note="o", packages=[PackagePick(flight_id="F1", hotel_id="H1", why="x")])
    plan = session.build_plan(draft, Usage(llm_model="t"))
    assert plan.packages == [] and "different dates" in plan.warnings[0]


FLEX_SCRIPT = [
    call("find_cheapest_dates", {"departure_airport": "TMP", "arrival_airport": "LPA"}, 1),
    call(
        "search_flights",
        {"departure_airports": ["TMP", "HEL"], "arrival_airports": ["LPA"], "depart_date": "2027-02-17"},
        2,
    ),
    call("search_hotels", {"location": "Maspalomas", "depart_date": "2027-02-17"}, 3),
    AIMessage(content="F1 + H1: 17 Feb is the cheapest date."),
]


FLEX_PLAN = call(
    "PlanDraft",
    {
        "summary": "17 Feb is cheapest.",
        "origin_note": "TMP.",
        "packages": [{"flight_id": "F1", "hotel_id": "H1", "why": "cheap dates"}],
    },
    5,
)


@respx.mock
def test_agent_picks_dates_from_calendar(tmp_path: Path) -> None:
    sent: list[dict[str, str]] = []

    def serpapi(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        sent.append(params)
        fixture = {"google_flights_calendar": "calendar.json", "google_flights": "flights.json"}
        return httpx.Response(200, json=load(fixture.get(params["engine"], "hotels.json")))

    respx.get(SERPAPI_URL).mock(side_effect=serpapi)
    respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))
    settings = Settings(groq_api_key="x", serpapi_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = ScriptedModel(script=FLEX_SCRIPT, plan=FLEX_PLAN)

    plan = plan_trip(make_request(**WINDOW), settings, model=model)

    assert "2027-02-17 -> 2027-02-24: ~1180 EUR (low)" in model.seen_tool_results[0]
    flights, hotels = sent[1], sent[2]
    assert (flights["outbound_date"], flights["return_date"]) == ("2027-02-17", "2027-02-24")
    assert (hotels["check_in_date"], hotels["check_out_date"]) == ("2027-02-17", "2027-02-24")
    package = plan.packages[0]
    assert package.flight.outbound_date == date(2027, 2, 17) and package.hotel.check_out == date(2027, 2, 24)
    assert plan.usage.serpapi_searches == 3
