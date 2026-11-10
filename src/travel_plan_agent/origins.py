"""Turn Finnish cities into departure airports, in order of preference.

The first city is home: its nearest airport comes first, with no extra cost. Any other cities
listed add their airports next. Helsinki (HEL) is added as a fallback when the cost of getting
there is known, because it has far more holiday routes. Getting to another airport costs extra
(train/bus). That rough cost per traveller is added to the package total, so a cheap flight from
further away only wins if it's really cheaper overall.
"""

from dataclasses import dataclass, field

HELSINKI = "HEL"


@dataclass(frozen=True)
class ResolvedOrigin:
    label: str
    airports: tuple[str, ...]
    transfer_per_person: dict[str, float] = field(default_factory=dict)
    transfer_note: dict[str, str] = field(default_factory=dict)

    def transfer_cost(self, airport: str, travellers: int) -> float:
        return self.transfer_per_person.get(airport, 0.0) * travellers


# Home airport and rough return-trip ground cost per traveller from the city to Helsinki Airport, in EUR.
# Edit these to match your own fares (children often travel cheaper), or set TRIP_TRANSFER_COSTS.
_FINNISH_CITIES: dict[str, tuple[str, float | None, str]] = {
    "tampere": ("TMP", 50.0, "VR train Tampere -> Helsinki Airport (~1h40), there and back"),
    "helsinki": ("HEL", None, ""),
    "vantaa": ("HEL", None, ""),
    "espoo": ("HEL", None, ""),
    "turku": ("TKU", 60.0, "bus/train Turku -> Helsinki Airport (~2h30), there and back"),
    "oulu": ("OUL", 120.0, "train Oulu -> Helsinki Airport (~6h), there and back"),
    "jyväskylä": ("JYV", 80.0, "train Jyväskylä -> Helsinki Airport (~3h30), there and back"),
    "jyvaskyla": ("JYV", 80.0, "train Jyväskylä -> Helsinki Airport (~3h30), there and back"),
    "kuopio": ("KUO", 100.0, "train Kuopio -> Helsinki Airport (~4h30), there and back"),
    "rovaniemi": ("RVN", 150.0, "night train Rovaniemi -> Helsinki, there and back"),
    "vaasa": ("VAA", 100.0, "train Vaasa -> Helsinki Airport (~4h), there and back"),
    "joensuu": ("JOE", 100.0, "train Joensuu -> Helsinki Airport (~4h30), there and back"),
}


def _airport_for(origin: str) -> tuple[str, float | None, str]:
    key = origin.strip().lower()
    if key in _FINNISH_CITIES:
        return _FINNISH_CITIES[key]
    code = origin.strip().upper()
    if len(code) == 3 and code.isalpha():
        return code, None, ""
    known = ", ".join(sorted({k.title() for k in _FINNISH_CITIES}))
    raise ValueError(f"Unknown origin '{origin}'. Use a 3-letter airport code or one of: {known}")


def resolve_origins(origins: list[str], transfer_costs: dict[str, float] | None = None) -> ResolvedOrigin:
    """Departure airports for the listed places. The first one is home.

    `transfer_costs` (per traveller, return trip, by airport code) overrides the built-in estimates.
    """
    if not origins:
        raise ValueError("at least one origin is needed")
    overrides = {k.upper(): v for k, v in (transfer_costs or {}).items()}
    home_airport, home_to_hel, hel_note = _airport_for(origins[0])

    airports: list[str] = []
    for origin in origins:
        airport = _airport_for(origin)[0]
        if airport not in airports:
            airports.append(airport)
    # HEL fallback only when we know what getting there from home costs; otherwise it would look free.
    if HELSINKI not in airports and (home_to_hel is not None or HELSINKI in overrides):
        airports.append(HELSINKI)

    costs: dict[str, float] = {}
    notes: dict[str, str] = {}
    for airport in airports:
        if airport == home_airport:
            continue
        if airport in overrides:
            costs[airport], notes[airport] = overrides[airport], "your estimate (TRIP_TRANSFER_COSTS)"
        elif airport == HELSINKI and home_to_hel is not None:
            costs[airport], notes[airport] = home_to_hel, hel_note
        else:
            notes[airport] = "no estimate; set TRIP_TRANSFER_COSTS to include it"
    label = " / ".join(o.strip().title() if len(o.strip()) > 3 else o.strip().upper() for o in origins)
    return ResolvedOrigin(label=label, airports=tuple(airports), transfer_per_person=costs, transfer_note=notes)


def resolve_origin(origin: str) -> ResolvedOrigin:
    return resolve_origins([origin])
