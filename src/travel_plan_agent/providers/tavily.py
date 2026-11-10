"""Web search through Tavily (https://tavily.com), used to find travel-agency package deals."""

from typing import Any

import httpx

from travel_plan_agent.cache import JsonCache
from travel_plan_agent.models import WebResult

TAVILY_URL = "https://api.tavily.com/search"

# Tour operators selling package holidays from Finland.
FINNISH_TRAVEL_AGENCIES = [
    "tjareborg.fi",
    "aurinkomatkat.fi",
    "tui.fi",
    "apollomatkat.fi",
    "detur.fi",
    "matkavekka.fi",
    "ebookers.fi",
    "momondo.fi",
]


class TavilyClient:
    def __init__(self, api_key: str, cache: JsonCache, http: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._cache = cache
        self._http = http or httpx.Client(timeout=60)
        self.live_calls = 0  # basic search = 1 Tavily credit

    def search(self, query: str, agencies_only: bool, id_prefix: str, max_results: int = 5) -> list[WebResult]:
        payload: dict[str, Any] = {"query": query, "max_results": max_results, "search_depth": "basic"}
        if agencies_only:
            payload["include_domains"] = FINNISH_TRAVEL_AGENCIES
        data = self._cache.get("tavily", payload)
        if data is None:
            response = self._http.post(TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {self._api_key}"})
            self.live_calls += 1
            response.raise_for_status()
            data = response.json()
            self._cache.set("tavily", payload, data)
        return [
            WebResult(
                id=f"{id_prefix}{i}",
                title=r.get("title", ""),
                url=r["url"],
                snippet=(r.get("content") or "")[:250],
            )
            for i, r in enumerate(data.get("results") or [], start=1)
            if r.get("url")
        ]
