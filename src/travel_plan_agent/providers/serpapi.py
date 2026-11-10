"""Google Flights and Google Hotels through SerpApi (https://serpapi.com)."""

from typing import Any, Literal

import httpx

from travel_plan_agent.cache import JsonCache
from travel_plan_agent.models import FlightLeg, FlightOption, HotelOption, TripRequest
from travel_plan_agent.passengers import flight_passengers, hotel_children_ages

SERPAPI_URL = "https://serpapi.com/search.json"

# SerpApi `stops`: 0 any, 1 nonstop, 2 one stop or fewer, 3 two stops or fewer.
_STOPS_PARAM = {0: 1, 1: 2, 2: 3}
# SerpApi hotel `rating`: 7 = 3.5+, 8 = 4.0+, 9 = 4.5+.
_RATING_PARAM = ((4.5, 9), (4.0, 8), (3.5, 7))
_SORT_PARAM = {"price": 3, "rating": 8, "reviews": 13}


class SerpApiError(RuntimeError):
    pass


class SerpApiClient:
    def __init__(
        self,
        api_key: str,
        cache: JsonCache,
        *,
        country: str = "fi",
        language: str = "en",
        http: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._cache = cache
        self._country = country
        self._language = language
        self._http = http or httpx.Client(timeout=60)
        self.live_calls = 0

    def _search(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run a search. Returns (response, came_from_cache)."""
        params = {**params, "gl": self._country, "hl": self._language}
        cached = self._cache.get("serpapi", params)
        if cached is not None:
            return cached, True
        response = self._http.get(SERPAPI_URL, params={**params, "api_key": self._api_key})
        self.live_calls += 1
        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise SerpApiError(f"SerpApi returned HTTP {response.status_code}") from exc
        if response.status_code != 200 or "error" in data:
            # "no results" is reported as an error by SerpApi; treat it as an empty result.
            message = str(data.get("error", f"HTTP {response.status_code}"))
            if "hasn't returned any results" in message:
                return {}, False
            raise SerpApiError(message)
        self._cache.set("serpapi", params, data)
        return data, False

    def flight_params(
        self, request: TripRequest, departure_ids: list[str], arrival_ids: list[str], max_stops: int | None
    ) -> dict[str, Any]:
        pax = flight_passengers(request.adults, request.children_ages)
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": ",".join(departure_ids),
            "arrival_id": ",".join(arrival_ids),
            "outbound_date": request.depart.isoformat(),
            "return_date": request.return_date.isoformat(),
            "type": 1,
            "adults": pax.adults,
            "currency": request.currency,
        }
        if pax.children:
            params["children"] = pax.children
        if pax.infants_on_lap:
            params["infant_on_lap"] = pax.infants_on_lap
        if max_stops is not None:
            params["stops"] = _STOPS_PARAM[max_stops]
        return params

    def search_flights(
        self,
        request: TripRequest,
        departure_ids: list[str],
        arrival_ids: list[str],
        max_stops: int | None,
        id_prefix: str,
    ) -> tuple[list[FlightOption], bool]:
        data, cached = self._search(self.flight_params(request, departure_ids, arrival_ids, max_stops))
        return parse_flights(data, request.currency, id_prefix), cached

    def hotel_params(
        self,
        request: TripRequest,
        location: str,
        max_price_per_night: int | None,
        min_rating: float | None,
        sort_by: Literal["price", "rating", "reviews"],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_hotels",
            "q": location,
            "check_in_date": request.depart.isoformat(),
            "check_out_date": request.return_date.isoformat(),
            "adults": request.adults,
            "currency": request.currency,
            "sort_by": _SORT_PARAM[sort_by],
        }
        if request.children_ages:
            ages = hotel_children_ages(request.children_ages)
            params["children"] = len(ages)
            params["children_ages"] = ",".join(str(a) for a in ages)
        if max_price_per_night:
            params["max_price"] = max_price_per_night
        if min_rating is not None:
            for threshold, code in _RATING_PARAM:
                if min_rating >= threshold:
                    params["rating"] = code
                    break
        return params

    def search_hotels(
        self,
        request: TripRequest,
        location: str,
        max_price_per_night: int | None,
        min_rating: float | None,
        sort_by: Literal["price", "rating", "reviews"],
        id_prefix: str,
    ) -> tuple[list[HotelOption], bool]:
        params = self.hotel_params(request, location, max_price_per_night, min_rating, sort_by)
        data, cached = self._search(params)
        return parse_hotels(data, location, request.nights, request.currency, id_prefix), cached


def parse_flights(data: dict[str, Any], currency: str, id_prefix: str) -> list[FlightOption]:
    """Read SerpApi Google Flights results. The price is the round-trip total for the whole party."""
    url = data.get("search_metadata", {}).get("google_flights_url")
    options: list[FlightOption] = []
    groups = [(True, data.get("best_flights") or []), (False, data.get("other_flights") or [])]
    for is_best, items in groups:
        for item in items:
            legs_raw = item.get("flights") or []
            price = item.get("price")
            if not legs_raw or price is None:
                continue
            legs = [
                FlightLeg(
                    from_airport=leg["departure_airport"]["id"],
                    to_airport=leg["arrival_airport"]["id"],
                    departs=leg["departure_airport"].get("time", ""),
                    arrives=leg["arrival_airport"].get("time", ""),
                    airline=leg.get("airline", "?"),
                    flight_number=leg.get("flight_number"),
                )
                for leg in legs_raw
            ]
            options.append(
                FlightOption(
                    id=f"{id_prefix}{len(options) + 1}",
                    price_total=float(price),
                    currency=currency,
                    departure_airport=legs[0].from_airport,
                    arrival_airport=legs[-1].to_airport,
                    airlines=sorted({leg.airline for leg in legs}),
                    stops=len(legs) - 1,
                    total_duration_min=item.get("total_duration"),
                    outbound_legs=legs,
                    booking_url=url,
                    is_best=is_best,
                )
            )
    return options


def parse_hotels(data: dict[str, Any], location: str, nights: int, currency: str, id_prefix: str) -> list[HotelOption]:
    fallback_url = data.get("search_metadata", {}).get("google_hotels_url")
    options: list[HotelOption] = []
    for prop in data.get("properties") or []:
        per_night = (prop.get("rate_per_night") or {}).get("extracted_lowest")
        total = (prop.get("total_rate") or {}).get("extracted_lowest")
        if total is None and per_night is not None:
            total = per_night * nights
        if total is None:
            continue
        sellers = [p["source"] for p in prop.get("prices") or [] if p.get("source")]
        options.append(
            HotelOption(
                id=f"{id_prefix}{len(options) + 1}",
                name=prop.get("name", "?"),
                location_query=location,
                kind=prop.get("type", "hotel"),
                total_price=float(total),
                price_per_night=float(per_night) if per_night is not None else None,
                currency=currency,
                rating=prop.get("overall_rating"),
                reviews=prop.get("reviews"),
                stars=prop.get("extracted_hotel_class"),
                amenities=list(prop.get("amenities") or [])[:8],
                sellers=sellers[:5],
                url=prop.get("link") or fallback_url,
            )
        )
    return options
