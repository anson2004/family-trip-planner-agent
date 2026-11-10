"""How each provider counts passengers by age."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlightPassengers:
    adults: int
    children: int  # ages 2-11
    infants_on_lap: int  # under 2


def flight_passengers(adults: int, children_ages: list[int]) -> FlightPassengers:
    """Airlines treat 12+ as adult fares, 2-11 as child fares, and under 2 as lap infants."""
    return FlightPassengers(
        adults=adults + sum(1 for a in children_ages if a >= 12),
        children=sum(1 for a in children_ages if 2 <= a <= 11),
        infants_on_lap=sum(1 for a in children_ages if a < 2),
    )


def hotel_children_ages(children_ages: list[int]) -> list[int]:
    """Google Hotels accepts child ages 1-17, so a baby counts as age 1."""
    return [max(1, a) for a in children_ages]
