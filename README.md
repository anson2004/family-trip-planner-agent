# travel-plan-agent

A LangChain agent (Groq LLM) that plans a holiday from Finland. It finds flight + hotel packages
for your party within a budget, and also looks for package holidays from Finnish travel agencies.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in the keys
```

| Key | Needed | Free plan |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | https://console.groq.com/keys |
| `SERPAPI_API_KEY` | yes (Google Flights + Google Hotels) | https://serpapi.com (small monthly quota) |
| `TAVILY_API_KEY` | optional (travel-agency packages) | https://app.tavily.com |
| `RESEND_API_KEY` + `EMAIL_TO` | optional (email the plan) | https://resend.com |

## Usage

Describe the trip with the `TRIP_*` values in `.env`, then run:

```bash
uv run travel-plan                      # plan the trip in .env
uv run travel-plan --trace              # also show every LLM call and its tokens
uv run travel-plan --json plan.json     # also save the plan as JSON
uv run travel-plan --env-file trips/turkey.env   # another trip file (needs the keys too)
```

| Setting | Example | Notes |
| --- | --- | --- |
| `TRIP_FROM` | `Tampere` or `Tampere,Turku` | Finnish city or airport code; comma-separate for several. The first is home (no transfer cost). HEL is added as a fallback with an estimated train cost (edit it in `src/travel_plan_agent/origins.py`) |
| `TRIP_TO` | `Gran Canaria` or `Gran Canaria,Tenerife,Madeira` | City, island, region or country; up to 5 comma-separated are compared in one plan. The agent picks the airports |
| `TRIP_TRANSFER_COSTS` | `TKU:40,HEL:50` | Optional return-trip ground cost per traveller to other departure airports |
| `TRIP_DEPART`, `TRIP_RETURN` | `2027-02-14` | YYYY-MM-DD |
| `TRIP_BUDGET` | `2000-3500` | Total EUR for flights + accommodation for everyone, or just a max like `3500` |
| `TRIP_ADULTS` | `2` | |
| `TRIP_CHILDREN_AGES` | `6,3` | Under-2s fly as lap infants, 12+ pay adult fares. Empty = no children |
| `TRIP_MAX_STOPS` | `1` | Optional |
| `TRIP_MIN_RATING` | `4.0` | Optional, minimum hotel guest rating |
| `TRIP_NOTES` | `near the beach, kids pool` | Optional wishes for the agent |

Environment variables override the file, e.g. `TRIP_TO=Turkey uv run travel-plan`.

### Email the plan

Set `RESEND_API_KEY` and `EMAIL_TO` (comma-separated) and every run emails the plan; `--no-email` skips it.
The default sender `onboarding@resend.dev` needs no domain setup, but Resend only delivers it to your own
Resend account's email. To send to other people, verify a domain in Resend and set `EMAIL_FROM`.

## How it works

```
CLI -> TripRequest (dates, party, budget: fixed, not chosen by the LLM)
    -> LangChain agent (Groq) with tools:
         search_flights        SerpApi Google Flights, all departure/arrival airports in one call
         search_hotels         SerpApi Google Hotels; each stay is shown with exact package totals
         search_travel_agencies Tavily, limited to Finnish tour operators (Tjäreborg, Aurinkomatkat, TUI, ...)
    -> PlanDraft (only IDs like F3 + H2) -> priced in Python -> TravelPlan table
```

The LLM never writes prices. Each result gets an ID, the agent returns the IDs it picked, and
`session.py` works out the totals from the real API data. Unknown IDs are dropped.

SerpApi results are cached in `~/.cache/travel-plan-agent` for 12 hours, so re-running costs nothing.
`MAX_SERPAPI_CALLS` caps the live searches per plan (default 4 + 2 per destination).

## Development

```bash
uv run pytest       # includes a full agent run with a scripted fake LLM, no keys needed
uv run mypy src tests
uv run ruff check src tests
```
