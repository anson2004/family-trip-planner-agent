from collections.abc import Callable
from typing import Any

import groq
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from travel_plan_agent.cache import JsonCache
from travel_plan_agent.config import Settings
from travel_plan_agent.models import LlmCall, PlanDraft, TravelPlan, TripRequest, Usage
from travel_plan_agent.origins import resolve_origins
from travel_plan_agent.providers.serpapi import SerpApiClient
from travel_plan_agent.providers.tavily import TavilyClient
from travel_plan_agent.session import TripSession
from travel_plan_agent.tools import build_tools

SYSTEM_PROMPT = """\
You are a family holiday planner for travellers leaving from Finland.
Find the best flight + accommodation combinations within the budget, using only the tools.

Trip (fixed, the tools already apply it):
- From: {origin_label}. Departure airports in order of preference: {airports}. {transfer_text}
- To: {destination}
- Dates: {depart} to {return_date} ({nights} nights)
- Travellers: {party}
- Budget for flights + accommodation, everyone together: {budget_min:.0f}-{budget_max:.0f} {currency}
{extra}
How to work:
1. For each destination: if it is a country or region, choose 1-3 main holiday airports there that
   have flights from Finland. If it is a city or island, use its airport(s).
2. Call search_flights with all departure airports and all destination airports together (up to 4
   destination airports per call; split into more calls only if there are more). Search again only
   if nothing useful came back (for example, allow more stops).
3. The first departure airport is the travellers' home and is preferred. Only recommend another
   departure airport when it is clearly cheaper overall, including the ground transfer.
   Explain that choice in origin_note.
4. Search hotels for each destination that has reasonable flights (one search per destination),
   in the area served by the best flights' arrival airport there. Prefer family-friendly
   places (pool, apartment/kitchen, beach). Room left for the hotel is roughly the budget maximum
   minus the flight price. Use max_price_per_night to stay within it.
5. Call search_travel_agencies once for package holidays from Finnish tour operators. Packages are
   often good value for families. Only mention a price that appears in the result text.
6. Hotel results already show exact package totals with the cheapest flights. Never do the sums
   yourself; the final totals are calculated by the system from the IDs you return.
7. Return 2-5 packages, best first, within budget wherever possible. Each flight must land near its hotel.
   {compare_text}
   Use only IDs the tools returned. Never invent prices, hotels or flights.
   You have {max_calls} flight/hotel searches in total, so plan them carefully.
8. Finish by calling the PlanDraft tool. Never write the final answer as text or a table.
Add a warning when something is uncertain (for example, flight prices change quickly, or the
child's ticket or bed rules need checking).
"""


def _system_prompt(request: TripRequest, session: TripSession, max_calls: int) -> str:
    origin = session.origin
    transfers = [
        f"Flying from {a} adds {origin.transfer_per_person[a]:.0f} {request.currency} per traveller "
        f"for ground travel ({note})."
        if a in origin.transfer_per_person
        else f"Ground travel to {a}: {note}."
        for a, note in origin.transfer_note.items()
    ]
    if len(request.destinations) > 1:
        compare_text = (
            "The travellers want to compare these destinations: include the best package for each "
            "destination where one fits the budget, and say in the summary which destination is the best value."
        )
    else:
        compare_text = ""
    extra = []
    if request.max_stops is not None:
        extra.append(f"- At most {request.max_stops} stop(s) each way.")
    if request.hotel_min_rating is not None:
        extra.append(f"- Hotel guest rating at least {request.hotel_min_rating}.")
    if request.notes:
        extra.append(f"- Traveller notes: {request.notes}")
    return SYSTEM_PROMPT.format(
        origin_label=origin.label,
        airports=", ".join(origin.airports),
        transfer_text=" ".join(transfers),
        destination="; ".join(request.destinations) + (" (compare them)" if len(request.destinations) > 1 else ""),
        compare_text=compare_text,
        depart=request.depart.isoformat(),
        return_date=request.return_date.isoformat(),
        nights=request.nights,
        party=request.party_label(),
        budget_min=request.budget_min,
        budget_max=request.budget_max,
        currency=request.currency,
        extra="\n".join(extra) + ("\n" if extra else ""),
        max_calls=max_calls,
    )


def plan_trip(
    request: TripRequest,
    settings: Settings,
    log: Callable[[str], None] = lambda _m: None,
    model: BaseChatModel | None = None,
) -> TravelPlan:
    """Run the agent and return a priced plan. `model` replaces the Groq model (useful for tests)."""
    cache = JsonCache(settings.cache_dir, settings.cache_ttl_hours)
    serpapi = SerpApiClient(
        settings.serpapi_api_key.get_secret_value(),
        cache,
        country=settings.google_country,
        language=settings.google_language,
    )
    tavily = TavilyClient(settings.tavily_api_key.get_secret_value(), cache) if settings.tavily_api_key else None
    session = TripSession(request=request, origin=resolve_origins(request.origins, request.transfer_costs), log=log)
    max_calls = settings.max_serpapi_calls or 4 + 2 * len(request.destinations)

    if model is None:
        model = ChatGroq(
            model_name=settings.groq_model,
            temperature=settings.groq_temperature,
            api_key=settings.groq_api_key,
            max_retries=3,
            # Less hidden reasoning = fewer output tokens. Only reasoning models accept this.
            reasoning_effort=settings.groq_reasoning_effort if _is_reasoning(settings.groq_model) else None,
        )
    agent: Any = create_agent(
        model,
        tools=build_tools(session, serpapi, tavily, max_calls),
        system_prompt=_system_prompt(request, session, max_calls),
        response_format=ToolStrategy(PlanDraft),
        # Groq rejects a reply that should have been a tool call but was text ("tool_use_failed").
        # Asking again usually works.
        middleware=[ModelRetryMiddleware(max_retries=2, retry_on=_is_tool_use_failed, on_failure="error")],
    )
    state: dict[str, Any] = {"messages": []}
    draft: PlanDraft | None = None
    try:
        # Stream so the conversation so far is kept even if the last step fails.
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "Plan this trip and return the final plan."}]},
            config={"recursion_limit": 40},
            stream_mode="values",
        ):
            state = chunk
        result = state.get("structured_response")
        draft = result if isinstance(result, PlanDraft) else None
    except Exception as exc:
        text = _failed_generation(exc) if _is_tool_use_failed(exc) else None
        if text is None:
            raise
        # The model wrote its final answer as text. All search results are still in the session,
        # so turn that text into a plan with one small call instead of starting over.
        log("Model answered in text; converting it into the plan")
        draft = _draft_from_text(model, text, session)
    if draft is None:
        raise RuntimeError("The agent finished without returning a plan.")
    usage = token_usage(state["messages"], llm_model=getattr(model, "model_name", type(model).__name__))
    usage.serpapi_searches = serpapi.live_calls
    usage.tavily_searches = tavily.live_calls if tavily else 0
    return session.build_plan(draft, usage)


def _groq_error(exc: BaseException) -> dict[str, Any]:
    if not isinstance(exc, groq.BadRequestError) or not isinstance(exc.body, dict):
        return {}
    inner = exc.body.get("error", exc.body)
    return inner if isinstance(inner, dict) else {}


def _is_tool_use_failed(exc: BaseException) -> bool:
    return _groq_error(exc).get("code") == "tool_use_failed"


def _failed_generation(exc: BaseException) -> str | None:
    text = _groq_error(exc).get("failed_generation")
    return text if isinstance(text, str) and text.strip() else None


_CONVERT_PROMPT = (
    "Convert the travel plan below into the PlanDraft format. Copy the flight IDs (F...), hotel IDs (H...) "
    "and web result IDs (W...) exactly as written; keep the reasons short. Valid IDs: "
    "flights {flights}; hotels {hotels}; web results {web}."
)


def _draft_from_text(model: BaseChatModel, text: str, session: TripSession) -> PlanDraft:
    structured = model.with_structured_output(PlanDraft)
    prompt = _CONVERT_PROMPT.format(
        flights=", ".join(session.flights) or "none",
        hotels=", ".join(session.hotels) or "none",
        web=", ".join(session.web) or "none",
    )
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            result = structured.invoke([SystemMessage(prompt), HumanMessage(text)])
        except Exception as exc:  # the conversion call can fail the same way; try again
            last_error = exc
            continue
        if isinstance(result, PlanDraft):
            return result
    raise RuntimeError(f"Could not turn the model's answer into a plan: {last_error}")


def _is_reasoning(model_name: str) -> bool:
    return model_name.startswith("openai/gpt-oss")


def token_usage(messages: list[BaseMessage], llm_model: str) -> Usage:
    """Add up the token counts the provider reported on each LLM response."""
    usage = Usage(llm_model=llm_model)
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        meta = msg.usage_metadata
        call = LlmCall(
            action=_describe(msg),
            input_tokens=meta["input_tokens"] if meta else 0,
            output_tokens=meta["output_tokens"] if meta else 0,
        )
        usage.calls.append(call)
        usage.llm_calls += 1
        usage.input_tokens += call.input_tokens
        usage.output_tokens += call.output_tokens
    return usage


def _describe(msg: AIMessage) -> str:
    """Short description of what the LLM asked for in this response."""
    if not msg.tool_calls:
        return "text reply"
    parts = []
    for tc in msg.tool_calls:
        if tc["name"] == PlanDraft.__name__:
            parts.append("final answer (PlanDraft)")
            continue
        args = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
        parts.append(f"{tc['name']}({args})")
    return "; ".join(parts)
