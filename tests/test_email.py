import json
from pathlib import Path

import httpx
import pytest
import respx

from tests.test_core import DATES, load, make_request
from travel_plan_agent.config import Settings
from travel_plan_agent.email_report import build_email, sample_plan
from travel_plan_agent.models import AgencyPick, PackagePick, PlanDraft, TravelPlan, Usage, WebResult
from travel_plan_agent.origins import resolve_origin
from travel_plan_agent.providers.resend import RESEND_URL, ResendError, send_email
from travel_plan_agent.providers.serpapi import parse_flights, parse_hotels
from travel_plan_agent.session import TripSession


def make_plan() -> TravelPlan:
    session = TripSession(request=make_request(), origin=resolve_origin("Tampere"))
    session.add_flights(parse_flights(load("flights.json"), "EUR", "x", DATES))
    session.add_hotels(parse_hotels(load("hotels.json"), "Gran Canaria", "EUR", "x", DATES))
    session.add_web([WebResult(id="x", title="Tjäreborg <Gran Canaria>", url="https://tjareborg.fi/gc", snippet="")])
    draft = PlanDraft(
        summary="Fly from Tampere & stay by the beach.",
        origin_note="TMP is within budget.",
        packages=[
            PackagePick(flight_id="F1", hotel_id="H1", why="pool <and> kids' club"),
            PackagePick(flight_id="F2", hotel_id="H2", why="cheapest"),
        ],
        agency_deals=[AgencyPick(result_id="W1", quoted_price="from 899 €/person", why="package")],
    )
    return session.build_plan(draft, Usage(llm_model="test"))


def test_build_email() -> None:
    email = build_email(make_plan())
    assert email.subject == "Trip plan: Tampere → Gran Canaria, 14 Feb – 21 Feb 2027 · from 2500 EUR"
    # Text from the LLM and from web pages is escaped, so it can't inject HTML.
    assert "pool &lt;and&gt; kids&#x27; club" in email.html and "<and>" not in email.html
    assert "Tjäreborg &lt;Gran Canaria&gt;" in email.html
    assert 'href="https://hotel.example/fbr"' in email.html
    assert "1. 2500 EUR (within budget)" in email.text
    assert "+ 150 transfer" in email.text  # HEL flight from Tampere
    assert "Hotel: https://hotel.example/fbr" in email.text


def test_sample_plan_email() -> None:
    email = build_email(sample_plan())
    assert "Sample Family Resort" in email.html and "test data" in email.text


@respx.mock
def test_send_email() -> None:
    route = respx.post(RESEND_URL).mock(return_value=httpx.Response(200, json={"id": "em_123"}))
    email = build_email(make_plan())
    assert send_email("re_key", "Trips <a@b.fi>", ["me@x.fi"], email) == "em_123"
    sent = route.calls[0].request
    assert sent.headers["Authorization"] == "Bearer re_key"
    body = json.loads(sent.content)
    assert body["to"] == ["me@x.fi"] and body["from"] == "Trips <a@b.fi>" and body["subject"] == email.subject


@respx.mock
def test_send_email_error() -> None:
    respx.post(RESEND_URL).mock(
        return_value=httpx.Response(403, json={"message": "You can only send testing emails to your own email"})
    )
    with pytest.raises(ResendError, match="403: You can only send testing emails"):
        send_email("re_key", "a@b.fi", ["other@x.fi"], build_email(make_plan()))


def test_email_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMAIL_TO", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=g\nSERPAPI_API_KEY=s\nRESEND_API_KEY=re_x\nEMAIL_TO=a@x.fi, b@x.fi\n")
    settings = Settings(_env_file=env)
    assert settings.email_to == ["a@x.fi", "b@x.fi"]
    assert settings.resend_api_key is not None and settings.email_from.endswith("<onboarding@resend.dev>")
