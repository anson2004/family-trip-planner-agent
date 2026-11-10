import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from travel_plan_agent.cache import JsonCache
from travel_plan_agent.config import Settings, TripSettings, parse_budget
from travel_plan_agent.models import (
    AgencyPick,
    BudgetStatus,
    PackagePick,
    PlanDraft,
    TripRequest,
    Usage,
    WebResult,
)
from travel_plan_agent.origins import resolve_origin, resolve_origins
from travel_plan_agent.passengers import flight_passengers, hotel_children_ages
from travel_plan_agent.providers.serpapi import SERPAPI_URL, SerpApiClient, parse_flights, parse_hotels
from travel_plan_agent.session import TripSession

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


def make_request(**overrides: Any) -> TripRequest:
    base: dict[str, Any] = dict(
        destinations=["Gran Canaria"],
        depart=date(2027, 2, 14),
        return_date=date(2027, 2, 21),
        budget_min=2000,
        budget_max=3000,
        adults=2,
        children_ages=[6],
    )
    return TripRequest(**{**base, **overrides})


def test_request_validation() -> None:
    assert make_request().nights == 7
    assert make_request().party_label() == "2 adults + 1 child (age 6)"
    with pytest.raises(ValueError):
        make_request(return_date=date(2027, 2, 10))
    with pytest.raises(ValueError):
        make_request(children_ages=[18])


def test_parse_budget() -> None:
    assert parse_budget("3000") == (0.0, 3000.0)
    assert parse_budget("2000-3500") == (2000.0, 3500.0)
    with pytest.raises(ValueError):
        parse_budget("lots")


def test_settings_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TRIP_FROM", "TRIP_CHILDREN_AGES", "TRIP_MAX_STOPS", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / "trip.env"
    env.write_text(
        "TRIP_TO=Turkey\nTRIP_DEPART=2027-06-20\nTRIP_RETURN=2027-06-27\nTRIP_BUDGET=4000\n"
        "TRIP_CHILDREN_AGES=9, 4\nTRIP_MAX_STOPS=\nGROQ_API_KEY=g\nSERPAPI_API_KEY=s\nTAVILY_API_KEY=\n"
    )
    request = TripSettings(_env_file=env).to_request()
    assert request.origins == ["Tampere"] and request.destinations == ["Turkey"]
    assert request.children_ages == [9, 4] and request.max_stops is None
    assert (request.budget_min, request.budget_max, request.nights) == (0, 4000, 7)
    settings = Settings(_env_file=env)
    assert settings.tavily_api_key is None  # empty value means "not set"


def test_passengers() -> None:
    pax = flight_passengers(2, [0, 5, 13])
    assert (pax.adults, pax.children, pax.infants_on_lap) == (3, 1, 1)
    assert hotel_children_ages([0, 5]) == [1, 5]


def test_origins() -> None:
    tampere = resolve_origin("tampere")
    assert tampere.airports == ("TMP", "HEL")
    assert tampere.transfer_cost("HEL", 3) == 150
    assert tampere.transfer_cost("TMP", 3) == 0
    assert resolve_origin("Helsinki").airports == ("HEL",)
    assert resolve_origin("tku").airports == ("TKU",)
    with pytest.raises(ValueError):
        resolve_origin("Nowhere town")


def test_multiple_origins() -> None:
    both = resolve_origins(["Tampere", "Turku"], {"tku": 40})
    assert both.airports == ("TMP", "TKU", "HEL")  # home first, HEL fallback last
    assert both.transfer_per_person == {"TKU": 40, "HEL": 50}
    assert both.label == "Tampere / Turku"
    unknown = resolve_origins(["Tampere", "Oulu"])
    assert "OUL" not in unknown.transfer_per_person and "TRIP_TRANSFER_COSTS" in unknown.transfer_note["OUL"]
    assert resolve_origins(["Tampere"], {"HEL": 30}).transfer_per_person == {"HEL": 30}


def test_multiple_destinations_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TRIP_FROM", "TRIP_TO", "TRIP_TRANSFER_COSTS"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / "trip.env"
    env.write_text(
        "TRIP_FROM=Tampere, Turku\nTRIP_TO=Gran Canaria, Tenerife ,Madeira\nTRIP_TRANSFER_COSTS=TKU:40, hel:45\n"
        "TRIP_DEPART=2027-02-14\nTRIP_RETURN=2027-02-21\nTRIP_BUDGET=3500\n"
    )
    request = TripSettings(_env_file=env).to_request()
    assert request.origins == ["Tampere", "Turku"]
    assert request.destinations == ["Gran Canaria", "Tenerife", "Madeira"]
    assert request.transfer_costs == {"TKU": 40, "HEL": 45}
    assert request.destination_label == "Gran Canaria / Tenerife / Madeira"
    env.write_text(env.read_text().replace("TKU:40", "TKU40"))
    with pytest.raises(ValueError, match="AIRPORT:COST"):
        TripSettings(_env_file=env)


def test_parse_flights() -> None:
    flights = parse_flights(load("flights.json"), "EUR", "F")
    assert len(flights) == 2  # the item without legs is skipped
    best = flights[0]
    assert best.is_best and best.price_total == 1450
    assert (best.departure_airport, best.arrival_airport, best.stops) == ("TMP", "LPA", 1)
    assert best.booking_url and "google.com/travel/flights" in best.booking_url


def test_parse_hotels_falls_back_to_nightly_rate() -> None:
    hotels = parse_hotels(load("hotels.json"), "Gran Canaria", nights=7, currency="EUR", id_prefix="H")
    assert [h.name for h in hotels] == ["Family Beach Resort", "Sea View Apartment"]
    assert hotels[0].total_price == 1050 and hotels[0].sellers == ["Booking.com", "Family Beach Resort"]
    assert hotels[1].total_price == 630  # 90 x 7 nights
    assert hotels[1].url and "google.com/travel/hotels" in hotels[1].url


@respx.mock
def test_serpapi_params_and_cache(tmp_path: Path) -> None:
    route = respx.get(SERPAPI_URL).mock(return_value=httpx.Response(200, json=load("flights.json")))
    client = SerpApiClient("key", JsonCache(tmp_path, 1))
    req = make_request(children_ages=[6, 1])
    flights, cached = client.search_flights(req, ["TMP", "HEL"], ["LPA"], max_stops=1, id_prefix="F")
    assert len(flights) == 2 and not cached
    params = route.calls[0].request.url.params
    assert params["departure_id"] == "TMP,HEL"
    assert params["adults"] == "2" and params["children"] == "1" and params["infant_on_lap"] == "1"
    assert params["stops"] == "2" and params["type"] == "1" and params["gl"] == "fi"

    _, cached = client.search_flights(req, ["TMP", "HEL"], ["LPA"], max_stops=1, id_prefix="F")
    assert cached and route.call_count == 1 and client.live_calls == 1


def test_hotel_params() -> None:
    client = SerpApiClient("key", JsonCache(Path("/nonexistent"), 1))
    params = client.hotel_params(make_request(children_ages=[6, 0]), "Maspalomas", 200, 4.2, "price")
    assert params["children"] == 2 and params["children_ages"] == "6,1"
    assert params["rating"] == 8 and params["max_price"] == 200 and params["sort_by"] == 3


def test_session_prices_and_builds_plan() -> None:
    session = TripSession(request=make_request(), origin=resolve_origin("Tampere"))
    flights = session.add_flights(parse_flights(load("flights.json"), "EUR", "x"))
    hotels = session.add_hotels(parse_hotels(load("hotels.json"), "Gran Canaria", 7, "EUR", "x"))
    session.add_web([WebResult(id="x", title="Tjäreborg Gran Canaria", url="https://tjareborg.fi/x", snippet="")])
    assert [f.id for f in flights] == ["F1", "F2"] and [h.id for h in hotels] == ["H1", "H2"]

    tmp_pkg = session.price(flights[0], hotels[0])  # TMP: no transfer
    assert tmp_pkg.total == 1450 + 1050 and tmp_pkg.budget_status is BudgetStatus.WITHIN
    hel_pkg = session.price(flights[1], hotels[1])  # HEL: 3 travellers x 50 transfer
    assert hel_pkg.ground_transfer == 150 and hel_pkg.total == 1100 + 630 + 150
    assert hel_pkg.budget_status is BudgetStatus.UNDER

    draft = PlanDraft(
        summary="s",
        origin_note="o",
        packages=[
            PackagePick(flight_id="F2", hotel_id="H2", why="cheap"),
            PackagePick(flight_id="F1", hotel_id="H1", why="best"),
            PackagePick(flight_id="F9", hotel_id="H1", why="made up"),
        ],
        agency_deals=[AgencyPick(result_id="W1", why="package"), AgencyPick(result_id="W7", why="made up")],
    )
    plan = session.build_plan(draft, Usage(llm_model="test"))
    assert [p.why for p in plan.packages] == ["best", "cheap"]  # within budget sorted first
    assert len(plan.agency_deals) == 1
    assert any("F9" in w for w in plan.warnings)
