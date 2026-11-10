from collections.abc import Callable
from typing import Any

import groq
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
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
- Dates: {dates_text}
- Travellers: {party}
- Budget for flights + accommodation, everyone together: {budget_min:.0f}-{budget_max:.0f} {currency}
{extra}
How to work:
{date_step}1. For each destination: if it is a country or region, choose 1-3 main holiday airports there that
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
8. When you are done, write your final recommendation as plain text: each package as its flight ID
   and hotel ID (e.g. "F3 + H2") with a short reason, any agency deals by their W ID, which departure
   airport you recommend and why, a short summary, and warnings.
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
    if request.flexible:
        dates_text = (
            f"flexible. A {request.nights}-day trip leaving any day from {request.depart} to "
            f"{request.latest_departure} (back by {request.return_date}). You choose the dates."
        )
        date_step = (
            "0. Dates are flexible: first call find_cheapest_dates for the main route of each destination\n"
            "   (home airport to that destination's main airport). Pick the 1-2 cheapest departure dates and\n"
            "   pass the same depart_date to search_flights and search_hotels, so flight and stay match.\n"
            "   Prefer one date that works for all destinations to save searches.\n"
        )
    else:
        dates_text = f"{request.depart} to {request.return_date} ({request.nights} nights), fixed"
        date_step = ""
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
        dates_text=dates_text,
        date_step=date_step,
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
    # Automatic cap: 4 + 2 per destination, plus 1 date search per destination when the dates are flexible.
    per_destination = 3 if request.flexible else 2
    max_calls = settings.max_serpapi_calls or 4 + per_destination * len(request.destinations)

    if model is None:
        model = ChatGroq(
            model_name=settings.groq_model,
            temperature=settings.groq_temperature,
            api_key=settings.groq_api_key,
            max_retries=3,
            # Less hidden reasoning = fewer output tokens. Only reasoning models accept this.
            reasoning_effort=settings.groq_reasoning_effort if _is_reasoning(settings.groq_model) else None,
        )
    # The agent researches with tools and finishes with a written recommendation. It is not forced
    # to answer through a tool: gpt-oss on Groq often breaks forced tool calls at the final step
    # ("tool_use_failed" / "output_parse_failed"). A separate small call turns the text into a plan.
    agent: Any = create_agent(
        model,
        tools=build_tools(session, serpapi, tavily, max_calls),
        system_prompt=_system_prompt(request, session, max_calls),
        middleware=[ModelRetryMiddleware(max_retries=2, retry_on=_is_generation_error, on_failure="error")],
    )
    state: dict[str, Any] = {"messages": []}
    try:
        # Stream so the conversation so far is kept even if a step fails.
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "Plan this trip and give your final recommendation."}]},
            config={"recursion_limit": 40},
            stream_mode="values",
        ):
            state = chunk
        answer = _final_text(state["messages"])
    except Exception as exc:
        # After the retries, Groq still couldn't read a reply. Its error carries what the model wrote,
        # and all search results are still in the session, so use that text instead of starting over.
        answer = _failed_generation(exc) if _is_generation_error(exc) else None
        if answer is None:
            raise
        log("Model reply was malformed; using its text")
    if not answer:
        raise RuntimeError("The agent finished without a recommendation.")
    log("Turning the recommendation into a plan")
    draft, extra_calls = _draft_from_text(model, answer, session)
    usage = token_usage(
        [*state["messages"], *extra_calls], llm_model=getattr(model, "model_name", type(model).__name__)
    )
    for call in usage.calls[len(usage.calls) - len(extra_calls) :]:
        call.action = "convert recommendation into plan"
    usage.serpapi_searches = serpapi.live_calls
    usage.tavily_searches = tavily.live_calls if tavily else 0
    return session.build_plan(draft, usage)


def _final_text(messages: list[BaseMessage]) -> str | None:
    """The agent's closing recommendation: the last AI message that isn't a tool call."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = msg.text if isinstance(msg.text, str) else str(msg.content)
            return text.strip() or None
    return None


def _groq_error(exc: BaseException) -> dict[str, Any]:
    if not isinstance(exc, groq.BadRequestError) or not isinstance(exc.body, dict):
        return {}
    inner = exc.body.get("error", exc.body)
    return inner if isinstance(inner, dict) else {}


# Groq's errors for a reply it couldn't read; asking the model again usually works.
_GENERATION_ERRORS = {"tool_use_failed", "output_parse_failed"}


def _is_generation_error(exc: BaseException) -> bool:
    return _groq_error(exc).get("code") in _GENERATION_ERRORS


def _failed_generation(exc: BaseException) -> str | None:
    text = _groq_error(exc).get("failed_generation")
    return text if isinstance(text, str) and text.strip() else None


_CONVERT_PROMPT = (
    "Convert the travel plan below into the PlanDraft format. Copy the flight IDs (F...), hotel IDs (H...) "
    "and web result IDs (W...) exactly as written; keep the reasons short. Valid IDs: "
    "flights {flights}; hotels {hotels}; web results {web}."
)


def _draft_from_text(model: BaseChatModel, text: str, session: TripSession) -> tuple[PlanDraft, list[AIMessage]]:
    """Turn the agent's written recommendation into a PlanDraft. Returns it with the raw replies (for usage)."""
    prompt = _CONVERT_PROMPT.format(
        flights=", ".join(session.flights) or "none",
        hotels=", ".join(session.hotels) or "none",
        web=", ".join(session.web) or "none",
    )
    # Groq's JSON-schema output is the most reliable for gpt-oss; function calling is the fallback.
    if isinstance(model, ChatGroq) and _is_reasoning(model.model_name):
        methods: list[str | None] = ["json_schema", "function_calling", "json_schema"]
    else:
        methods = [None, None, None]  # the model's default structured-output method
    raw_replies: list[AIMessage] = []
    last_error: object = None
    for method in methods:
        kwargs: dict[str, Any] = {"method": method} if method else {}
        structured = model.with_structured_output(PlanDraft, include_raw=True, **kwargs)
        try:
            result: Any = structured.invoke([SystemMessage(prompt), HumanMessage(text)])
        except Exception as exc:  # the conversion call can fail the same way; try again
            last_error = exc
            continue
        if isinstance(result.get("raw"), AIMessage):
            raw_replies.append(result["raw"])
        if isinstance(result.get("parsed"), PlanDraft):
            return result["parsed"], raw_replies
        last_error = result.get("parsing_error")
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
        return "final recommendation (text)"
    parts = []
    for tc in msg.tool_calls:
        args = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
        parts.append(f"{tc['name']}({args})")
    return "; ".join(parts)
