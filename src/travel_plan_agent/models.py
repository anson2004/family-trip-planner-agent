from datetime import date
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class TripRequest(BaseModel):
    """What the traveller asked for. Dates and party are fixed here, never chosen by the LLM."""

    origins: list[str] = Field(default_factory=lambda: ["Tampere"], min_length=1, max_length=4)
    destinations: list[str] = Field(min_length=1, max_length=5)
    # Return-trip ground cost per traveller to reach a departure airport, overriding the estimates.
    transfer_costs: dict[str, float] = Field(default_factory=dict)
    depart: date
    return_date: date
    budget_min: float = Field(ge=0)
    budget_max: float = Field(gt=0)
    adults: int = Field(default=2, ge=1, le=9)
    children_ages: list[int] = Field(default_factory=list)
    currency: str = "EUR"
    max_stops: int | None = Field(default=None, ge=0, le=2)
    hotel_min_rating: float | None = Field(default=None, ge=0, le=5)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.return_date <= self.depart:
            raise ValueError("return date must be after departure date")
        if self.budget_min > self.budget_max:
            raise ValueError("budget_min must not exceed budget_max")
        if any(a < 0 or a > 17 for a in self.children_ages):
            raise ValueError("child ages must be between 0 and 17")
        if self.adults + len(self.children_ages) > 9:
            raise ValueError("at most 9 travellers are supported")
        return self

    @property
    def origin_label(self) -> str:
        return " / ".join(self.origins)

    @property
    def destination_label(self) -> str:
        return " / ".join(self.destinations)

    @property
    def nights(self) -> int:
        return (self.return_date - self.depart).days

    @property
    def travellers(self) -> int:
        return self.adults + len(self.children_ages)

    def party_label(self) -> str:
        label = f"{self.adults} adult{'s' if self.adults != 1 else ''}"
        if self.children_ages:
            ages = ", ".join(str(a) for a in self.children_ages)
            label += f" + {len(self.children_ages)} child{'ren' if len(self.children_ages) != 1 else ''} (age {ages})"
        return label


class FlightLeg(BaseModel):
    from_airport: str
    to_airport: str
    departs: str
    arrives: str
    airline: str
    flight_number: str | None = None


class FlightOption(BaseModel):
    id: str
    price_total: float
    currency: str
    departure_airport: str
    arrival_airport: str
    airlines: list[str]
    stops: int
    total_duration_min: int | None
    outbound_legs: list[FlightLeg]
    booking_url: str | None
    is_best: bool = False


class HotelOption(BaseModel):
    id: str
    name: str
    location_query: str
    kind: str
    total_price: float
    price_per_night: float | None
    currency: str
    rating: float | None
    reviews: int | None
    stars: int | None
    amenities: list[str] = Field(default_factory=list)
    sellers: list[str] = Field(default_factory=list)
    url: str | None


class WebResult(BaseModel):
    id: str
    title: str
    url: str
    snippet: str


# What the LLM returns: references to tool results by ID, never prices.
class PackagePick(BaseModel):
    flight_id: str = Field(description="ID of a flight from search_flights, e.g. 'F3'")
    hotel_id: str = Field(description="ID of a hotel from search_hotels, e.g. 'H2'")
    why: str = Field(description="One or two sentences on why this combination fits the travellers")


class AgencyPick(BaseModel):
    result_id: str = Field(description="ID of a result from search_travel_agencies, e.g. 'W1'")
    quoted_price: str | None = Field(
        default=None, description="Price exactly as shown in the snippet, or null if none was shown"
    )
    why: str


class PlanDraft(BaseModel):
    """The final answer: picked packages and deals, referenced by the IDs returned by the tools."""

    summary: str = Field(description="Short overview of the recommendation for the travellers")
    origin_note: str = Field(description="Which departure airport is recommended and why")
    packages: list[PackagePick] = Field(description="2 to 5 flight + hotel combinations, best first")
    agency_deals: list[AgencyPick] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# Resolved, priced plan shown to the user.
class BudgetStatus(StrEnum):
    WITHIN = "within budget"
    UNDER = "under budget"
    OVER = "over budget"


class PricedPackage(BaseModel):
    flight: FlightOption
    hotel: HotelOption
    ground_transfer: float
    total: float
    budget_status: BudgetStatus
    why: str


class AgencyDeal(BaseModel):
    result: WebResult
    quoted_price: str | None
    why: str


class LlmCall(BaseModel):
    """One request to the LLM: what it decided to do, and the tokens it cost."""

    action: str
    input_tokens: int = 0
    output_tokens: int = 0


class Usage(BaseModel):
    """What one plan cost: LLM tokens and live API searches (cache hits are free and not counted)."""

    llm_model: str
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    serpapi_searches: int = 0
    tavily_searches: int = 0
    calls: list[LlmCall] = Field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TravelPlan(BaseModel):
    request: TripRequest
    summary: str
    origin_note: str
    packages: list[PricedPackage]
    agency_deals: list[AgencyDeal]
    warnings: list[str]
    usage: Usage
