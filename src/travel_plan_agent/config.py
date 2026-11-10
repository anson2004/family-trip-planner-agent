from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from travel_plan_agent.models import TripRequest

# Empty values like `TAVILY_API_KEY=` count as "not set", so defaults apply.
_ENV = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)


class Settings(BaseSettings):
    """API keys and agent settings, read from environment variables or a .env file."""

    model_config = _ENV

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"
    groq_temperature: float = 0.0
    # For gpt-oss models: "low" keeps the hidden reasoning (billed as output tokens) short.
    groq_reasoning_effort: str = "low"

    serpapi_api_key: SecretStr
    tavily_api_key: SecretStr | None = None

    # SerpApi's free plan has a small monthly quota; cap how many searches one plan may use.
    # Empty = automatic: 4 + 2 per destination (6 for one destination, 10 for three).
    max_serpapi_calls: int | None = Field(default=None, ge=1)
    cache_ttl_hours: float = 12.0
    cache_dir: Path = Path.home() / ".cache" / "travel-plan-agent"

    # Google market settings: Finnish market, English text.
    google_country: str = "fi"
    google_language: str = "en"

    # Optional: email the plan through Resend. Sent when RESEND_API_KEY and EMAIL_TO are both set.
    resend_api_key: SecretStr | None = None
    email_to: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # onboarding@resend.dev works without your own domain, but only to your Resend account's address.
    email_from: str = "Trip Planner <onboarding@resend.dev>"

    @field_validator("email_to", mode="before")
    @classmethod
    def _split_emails(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [e.strip() for e in value.split(",") if e.strip()]
        return value


def parse_budget(value: str) -> tuple[float, float]:
    """Accept '3000' (maximum only) or '2000-3500' (a range)."""
    parts = [p.strip() for p in value.replace("€", "").split("-")]
    try:
        if len(parts) == 1:
            return 0.0, float(parts[0])
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    raise ValueError("use a maximum like 3000 or a range like 2000-3500")


class TripSettings(BaseSettings):
    """The trip to plan, read from TRIP_* environment variables or a .env file."""

    model_config = _ENV

    origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["Tampere"], validation_alias="TRIP_FROM")
    destinations: Annotated[list[str], NoDecode] = Field(validation_alias="TRIP_TO")
    transfer_costs: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=dict, validation_alias="TRIP_TRANSFER_COSTS"
    )
    depart: date = Field(validation_alias="TRIP_DEPART")
    return_date: date = Field(validation_alias="TRIP_RETURN")
    # Optional trip length. When set, TRIP_DEPART/TRIP_RETURN are the search window.
    trip_days: int | None = Field(default=None, validation_alias="TRIP_DAYS")
    budget: str = Field(validation_alias="TRIP_BUDGET")
    adults: int = Field(default=2, validation_alias="TRIP_ADULTS")
    children_ages: Annotated[list[int], NoDecode] = Field(default_factory=list, validation_alias="TRIP_CHILDREN_AGES")
    max_stops: int | None = Field(default=None, validation_alias="TRIP_MAX_STOPS")
    min_rating: float | None = Field(default=None, validation_alias="TRIP_MIN_RATING")
    notes: str | None = Field(default=None, validation_alias="TRIP_NOTES")

    @field_validator("origins", "destinations", mode="before")
    @classmethod
    def _split_places(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @field_validator("transfer_costs", mode="before")
    @classmethod
    def _parse_costs(cls, value: Any) -> Any:
        """'TKU:40, HEL:50' -> {"TKU": 40.0, "HEL": 50.0}"""
        if not isinstance(value, str):
            return value
        costs: dict[str, float] = {}
        for item in filter(None, (i.strip() for i in value.split(","))):
            airport, sep, cost = item.partition(":")
            if not sep:
                raise ValueError("use AIRPORT:COST pairs, e.g. TKU:40,HEL:50")
            costs[airport.strip().upper()] = float(cost)
        return costs

    @field_validator("children_ages", mode="before")
    @classmethod
    def _split_ages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [int(a) for a in value.replace(" ", "").split(",") if a]
        return value

    @field_validator("budget")
    @classmethod
    def _check_budget(cls, value: str) -> str:
        parse_budget(value)
        return value

    def to_request(self) -> TripRequest:
        budget_min, budget_max = parse_budget(self.budget)
        return TripRequest(
            origins=self.origins,
            destinations=self.destinations,
            transfer_costs=self.transfer_costs,
            depart=self.depart,
            return_date=self.return_date,
            trip_days=self.trip_days,
            budget_min=budget_min,
            budget_max=budget_max,
            adults=self.adults,
            children_ages=self.children_ages,
            max_stops=self.max_stops,
            hotel_min_rating=self.min_rating,
            notes=self.notes,
        )
