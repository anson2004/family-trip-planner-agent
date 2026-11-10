"""Runs the whole LangChain agent loop with a scripted fake LLM and mocked HTTP. No API keys needed."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import groq
import httpx
import pytest
import respx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from tests.test_core import load, make_request
from travel_plan_agent.agent import plan_trip
from travel_plan_agent.config import Settings
from travel_plan_agent.models import BudgetStatus
from travel_plan_agent.providers.serpapi import SERPAPI_URL
from travel_plan_agent.providers.tavily import TAVILY_URL


def call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}],
        usage_metadata={"input_tokens": 1000 * i, "output_tokens": 50, "total_tokens": 1000 * i + 50},
    )


SCRIPT = [
    call("search_flights", {"departure_airports": ["TMP", "HEL"], "arrival_airports": ["LPA"]}, 1),
    call("search_hotels", {"location": "Maspalomas, Gran Canaria", "max_price_per_night": 220}, 2),
    call("search_travel_agencies", {"query": "Gran Canaria family package from Tampere"}, 3),
    call(
        "PlanDraft",
        {
            "summary": "Fly direct from Tampere and stay at a family resort.",
            "origin_note": "TMP preferred and within budget.",
            "packages": [
                {"flight_id": "F2", "hotel_id": "H2", "why": "cheapest"},
                {"flight_id": "F1", "hotel_id": "H1", "why": "family resort, from Tampere"},
            ],
            "agency_deals": [{"result_id": "W1", "quoted_price": "from 899 €/person", "why": "package"}],
        },
        4,
    ),
]


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order and records what the tools sent back."""

    script: list[AIMessage]
    seen_tool_results: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.seen_tool_results = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        return ChatResult(generations=[ChatGeneration(message=self.script[step])])


@respx.mock
def test_agent_end_to_end(tmp_path: Path) -> None:
    def serpapi(request: httpx.Request) -> httpx.Response:
        engine = request.url.params["engine"]
        return httpx.Response(200, json=load("flights.json" if engine == "google_flights" else "hotels.json"))

    respx.get(SERPAPI_URL).mock(side_effect=serpapi)
    respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"title": "Tjäreborg Gran Canaria", "url": "https://tjareborg.fi/gc", "content": "x"}]},
        )
    )
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = ScriptedModel(script=SCRIPT)
    logs: list[str] = []
    log: Callable[[str], None] = logs.append

    plan = plan_trip(make_request(), settings, log=log, model=model)

    assert [p.why for p in plan.packages] == ["family resort, from Tampere", "cheapest"]
    best = plan.packages[0]
    assert best.total == 2500 and best.budget_status is BudgetStatus.WITHIN and best.ground_transfer == 0
    assert plan.packages[1].ground_transfer == 150  # HEL flight from Tampere: 3 x 50
    assert plan.agency_deals[0].result.url == "https://tjareborg.fi/gc"
    u = plan.usage
    assert (u.serpapi_searches, u.tavily_searches, u.llm_calls) == (2, 1, 4)
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (10000, 200, 10200)
    assert any("with F1 (TMP): 2500 total, within budget" in r for r in model.seen_tool_results)
    assert logs[0] == "Flights TMP,HEL -> LPA"
    assert [c.input_tokens for c in u.calls] == [1000, 2000, 3000, 4000]
    assert u.calls[0].action == "search_flights(departure_airports=['TMP', 'HEL'], arrival_airports=['LPA'])"
    assert u.calls[-1].action == "final answer (PlanDraft)"


def test_tools_are_typed() -> None:
    from travel_plan_agent.origins import resolve_origin
    from travel_plan_agent.session import TripSession
    from travel_plan_agent.tools import build_tools

    session = TripSession(request=make_request(), origin=resolve_origin("Tampere"))
    tools: list[BaseTool] = build_tools(session, serpapi=None, tavily=None, max_calls=1)  # type: ignore[arg-type]
    schema_model = tools[0].tool_call_schema
    assert isinstance(schema_model, type) and issubclass(schema_model, BaseModel)
    schema = schema_model.model_json_schema()
    assert schema["properties"]["departure_airports"]["type"] == "array"
    assert "IATA" in schema["properties"]["departure_airports"]["description"]


def _tool_use_failed() -> groq.BadRequestError:
    """The error Groq returns when the model writes text where a tool call was required."""
    body = {
        "error": {
            "message": "Tool choice is required, but model did not call a tool",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
            "failed_generation": "**PlanDraft**\n| 1 | **F1** | **H1** | resort |\n| 2 | **F2** | **H2** | cheap |",
        }
    }
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return groq.BadRequestError("tool_use_failed", response=response, body=body)


class FlakyFinalModel(ScriptedModel):
    """Follows the script, but the final answer fails `fail_times` times, like gpt-oss on Groq did."""

    fail_times: int
    final_attempts: int = 0
    conversions: int = 0

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        if isinstance(messages[0].content, str) and messages[0].content.startswith("Convert the travel plan"):
            self.conversions += 1
            return ChatResult(generations=[ChatGeneration(message=SCRIPT[-1])])
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        if step == len(self.script) - 1:
            self.final_attempts += 1
            if self.final_attempts <= self.fail_times:
                raise _tool_use_failed()
        return super()._generate(messages, stop, run_manager, **kw)


def _mock_apis() -> None:
    respx.get(SERPAPI_URL).mock(
        side_effect=lambda r: httpx.Response(
            200, json=load("flights.json" if r.url.params["engine"] == "google_flights" else "hotels.json")
        )
    )
    respx.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={"results": []}))


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain.agents.middleware.model_retry.time.sleep", lambda _s: None)


@respx.mock
def test_retries_when_final_answer_is_text(tmp_path: Path, no_sleep: None) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = FlakyFinalModel(script=SCRIPT, fail_times=1)
    plan = plan_trip(make_request(), settings, model=model)
    assert model.final_attempts == 2 and model.conversions == 0
    assert [p.why for p in plan.packages] == ["family resort, from Tampere", "cheapest"]


@respx.mock
def test_converts_text_answer_when_retries_fail(tmp_path: Path, no_sleep: None) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = FlakyFinalModel(script=SCRIPT, fail_times=99)
    logs: list[str] = []
    plan = plan_trip(make_request(), settings, log=logs.append, model=model)
    assert model.final_attempts == 3  # first try + 2 retries
    assert model.conversions == 1
    assert "Model answered in text; converting it into the plan" in logs
    assert len(plan.packages) == 2 and plan.packages[0].total == 2500  # priced from the saved search results
    assert plan.usage.serpapi_searches == 2  # nothing was searched again
