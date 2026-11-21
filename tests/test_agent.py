"""Runs the whole LangChain agent loop with a scripted fake LLM and mocked HTTP. No API keys needed."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import groq
import httpx
import pytest
import respx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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
    AIMessage(
        content="Recommend F1 + H1 (family resort, from Tampere) and F2 + H2 (cheapest). Agency: W1.",
        usage_metadata={"input_tokens": 4000, "output_tokens": 50, "total_tokens": 4050},
    ),
]

# What the conversion call returns: the recommendation as a PlanDraft.
PLAN = call(
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
    5,
)


def is_conversion(messages: list[BaseMessage]) -> bool:
    first = messages[0].content
    return isinstance(first, str) and first.startswith("Convert the travel plan")


class ScriptedModel(BaseChatModel):
    """Returns the scripted messages in order and records what the tools sent back."""

    script: list[AIMessage]
    plan: AIMessage = PLAN
    seen_tool_results: list[str] = []
    conversions: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        if is_conversion(messages):
            self.conversions += 1
            return ChatResult(generations=[ChatGeneration(message=self.plan)])
        self.seen_tool_results = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        return ChatResult(generations=[ChatGeneration(message=self.script[step])])


@respx.mock
def test_agent_end_to_end(tmp_path: Path) -> None:
    def serpapi(request: httpx.Request) -> httpx.Response:
        engine = request.url.params["engine"]
        return httpx.Response(200, json=load("flights.json" if engine == "google_flights" else "hotels.json"))

    respx.get(SERPAPI_URL).mock(side_effect=serpapi)
    tavily = respx.post(TAVILY_URL).mock(
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

    assert [p.why for p in plan.packages] == ["cheapest", "family resort, from Tampere"]  # cheapest first
    best = plan.packages[1]
    assert best.total == 2500 and best.budget_status is BudgetStatus.WITHIN and best.ground_transfer == 0
    assert plan.packages[0].ground_transfer == 150  # HEL flight from Tampere: 3 x 50
    assert plan.agency_deals[0].result.url == "https://tjareborg.fi/gc"
    u = plan.usage
    # Tavily: the package search the system runs up front, plus the one the scripted agent asked for.
    assert (u.serpapi_searches, u.tavily_searches, u.llm_calls) == (2, 2, 5)  # 4 agent steps + 1 conversion
    first_query = json.loads(tavily.calls[0].request.content)["query"]
    assert first_query == "Gran Canaria family package holiday from Tampere February 2027"
    assert logs[0] == f"Web search: {first_query} (1 result(s))"
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (15000, 250, 15250)
    assert model.conversions == 1
    assert any("with F1 (TMP): 2500 total, within budget" in r for r in model.seen_tool_results)
    assert logs[1] == "Flights TMP,HEL -> LPA (2027-02-14)"
    assert [c.input_tokens for c in u.calls] == [1000, 2000, 3000, 4000, 5000]
    assert u.calls[0].action == "search_flights(departure_airports=['TMP', 'HEL'], arrival_airports=['LPA'])"
    assert u.calls[-2].action == "final recommendation (text)"
    assert u.calls[-1].action == "convert recommendation into plan"


def test_tools_are_typed() -> None:
    from travel_plan_agent.origins import resolve_origin
    from travel_plan_agent.session import TripSession
    from travel_plan_agent.tools import build_tools

    session = TripSession(request=make_request(), origin=resolve_origin("Tampere"))
    tools: list[BaseTool] = build_tools(session, serpapi=None, tavily=None)  # type: ignore[arg-type]
    schema_model = tools[0].tool_call_schema
    assert isinstance(schema_model, type) and issubclass(schema_model, BaseModel)
    schema = schema_model.model_json_schema()
    assert schema["properties"]["departure_airports"]["type"] == "array"
    assert "IATA" in schema["properties"]["departure_airports"]["description"]


FINAL_TABLE = "**PlanDraft**\n| 1 | **F1** | **H1** | resort |\n| 2 | **F2** | **H2** | cheap |"


def _generation_error(code: str = "tool_use_failed", text: str = FINAL_TABLE) -> groq.BadRequestError:
    """The error Groq returns when the model writes text where a tool call was required."""
    body = {
        "error": {
            "message": "Tool choice is required, but model did not call a tool",
            "type": "invalid_request_error",
            "code": code,
            "failed_generation": text,
        }
    }
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return groq.BadRequestError("tool_use_failed", response=response, body=body)


class FlakyFinalModel(ScriptedModel):
    """Follows the script, but the final answer fails `fail_times` times, like gpt-oss on Groq did."""

    fail_times: int
    error_code: str = "tool_use_failed"
    final_attempts: int = 0

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        if is_conversion(messages):
            return super()._generate(messages, stop, run_manager, **kw)
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        if step == len(self.script) - 1:
            self.final_attempts += 1
            if self.final_attempts <= self.fail_times:
                raise _generation_error(self.error_code)
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
@pytest.mark.parametrize("code", ["tool_use_failed", "output_parse_failed"])
def test_retries_when_groq_cannot_read_the_reply(tmp_path: Path, no_sleep: None, code: str) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = FlakyFinalModel(script=SCRIPT, fail_times=1, error_code=code)
    plan = plan_trip(make_request(), settings, model=model)
    assert model.final_attempts == 2 and model.conversions == 1
    assert [p.why for p in plan.packages] == ["cheapest", "family resort, from Tampere"]


@respx.mock
@pytest.mark.parametrize("code", ["tool_use_failed", "output_parse_failed"])
def test_uses_failed_generation_when_retries_fail(tmp_path: Path, no_sleep: None, code: str) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = FlakyFinalModel(script=SCRIPT, fail_times=99, error_code=code)
    logs: list[str] = []
    plan = plan_trip(make_request(), settings, log=logs.append, model=model)
    assert model.final_attempts == 3  # first try + 2 retries
    assert model.conversions == 1
    assert "Model reply was malformed; using its text" in logs
    assert [p.total for p in plan.packages] == [1880, 2500]  # priced from the saved search results
    assert plan.usage.serpapi_searches == 2  # nothing was searched again


def test_other_errors_are_not_hidden(tmp_path: Path) -> None:
    class Broken(ScriptedModel):
        def _generate(
            self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
        ) -> ChatResult:
            raise RuntimeError("network down")

    settings = Settings(groq_api_key="x", serpapi_api_key="x", cache_dir=tmp_path, _env_file=None)
    with pytest.raises(RuntimeError, match="network down"):
        plan_trip(make_request(), settings, model=Broken(script=SCRIPT))


class AskedAgainModel(ScriptedModel):
    """Answers the "stop searching" request with a recommendation naming only F1 + H1."""

    asked_again: int = 0
    history_before_ask: list[BaseMessage] = []

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage) and str(last.content).startswith("Stop searching"):
            self.asked_again += 1
            self.history_before_ask = messages[:-1]
            reply = AIMessage(
                content="F1 + H1: family resort, direct from Tampere.",
                usage_metadata={"input_tokens": 7000, "output_tokens": 40, "total_tokens": 7040},
            )
            return ChatResult(generations=[ChatGeneration(message=reply)])
        return super()._generate(messages, stop, run_manager, **kw)


class BrokenSearchModel(AskedAgainModel):
    """The web-search step keeps failing with a broken tool call that names no packages."""

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        if not is_conversion(messages) and isinstance(messages[-1], ToolMessage) and step == 2:
            raise _generation_error(text='{"name": "search_travel_agencies", "arguments": {"query": ')
        return super()._generate(messages, stop, run_manager, **kw)


@respx.mock
def test_broken_search_call_is_not_used_as_the_answer(tmp_path: Path, no_sleep: None) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", tavily_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = BrokenSearchModel(script=SCRIPT)
    logs: list[str] = []

    plan = plan_trip(make_request(), settings, log=logs.append, model=model)

    assert "Model reply was malformed; asking for its recommendation" in logs
    assert model.asked_again == 1 and model.conversions == 1
    # The conversion returns F2+H2, F1+H1 and W1, but the recommendation only named F1 + H1.
    # Only one package was recommended, so the system adds the next cheapest one (cheaper, so it comes first).
    assert [p.why for p in plan.packages] == [
        "Added by the system: the next cheapest combination found.",
        "family resort, from Tampere",
    ]
    assert plan.agency_deals == []
    assert "Dropped F2+H2: not in the agent's recommendation." in plan.warnings
    actions = [c.action for c in plan.usage.calls]
    assert actions[2:] == [
        "agent step (failed, tokens not reported)",
        "final recommendation (asked again)",
        "convert recommendation into plan",
    ]


class EndlessSearchModel(AskedAgainModel):
    """Searches flights and hotels, then repeats the flight search until the step limit stops it."""

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        last = messages[-1]
        asking = isinstance(last, HumanMessage) and str(last.content).startswith("Stop searching")
        if is_conversion(messages) or asking or step < 2:
            return super()._generate(messages, stop, run_manager, **kw)
        again = call("search_flights", {"departure_airports": ["TMP"], "arrival_airports": ["LPA"]}, step + 1)
        return ChatResult(generations=[ChatGeneration(message=again)])


@respx.mock
def test_step_limit_asks_for_the_recommendation(tmp_path: Path) -> None:
    _mock_apis()
    settings = Settings(groq_api_key="x", serpapi_api_key="x", cache_dir=tmp_path, _env_file=None)
    model = EndlessSearchModel(script=SCRIPT)
    logs: list[str] = []

    plan = plan_trip(make_request(), settings, log=logs.append, model=model)

    assert "Agent ran out of steps; asking for its recommendation" in logs
    assert model.asked_again == 1
    last = model.history_before_ask[-1]
    assert not (isinstance(last, AIMessage) and last.tool_calls)  # no tool call left without its result
    # Only one package was recommended, so the system adds the next cheapest one (cheaper, so it comes first).
    assert [p.why for p in plan.packages] == [
        "Added by the system: the next cheapest combination found.",
        "family resort, from Tampere",
    ]
    assert plan.usage.serpapi_searches == 3  # TMP+HEL, hotels, TMP only; the repeats came from the cache
