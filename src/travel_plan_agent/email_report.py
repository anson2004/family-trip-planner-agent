"""Turn a TravelPlan into an email (subject, HTML and plain text)."""

from dataclasses import dataclass
from datetime import date
from html import escape

from travel_plan_agent.models import (
    BudgetStatus,
    FlightLeg,
    FlightOption,
    HotelOption,
    PricedPackage,
    TravelPlan,
    TripRequest,
    Usage,
)

_STATUS_COLOR = {BudgetStatus.WITHIN: "#1a7f37", BudgetStatus.UNDER: "#0969da", BudgetStatus.OVER: "#cf222e"}
_CELL = "padding:8px;border:1px solid #d0d7de;vertical-align:top;text-align:left"


@dataclass(frozen=True)
class EmailContent:
    subject: str
    html: str
    text: str


def _stops(p: PricedPackage) -> str:
    return "nonstop" if p.flight.stops == 0 else f"{p.flight.stops} stop(s)"


def _link(url: str | None, label: str) -> str:
    return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>' if url else ""


def build_email(plan: TravelPlan) -> EmailContent:
    r, cur = plan.request, plan.request.currency
    title = f"{r.origin_label} → {r.destination_label}, {r.depart:%d %b} – {r.return_date:%d %b %Y}"
    best = next((p for p in plan.packages if p.budget_status is BudgetStatus.WITHIN), None)
    subject = f"Trip plan: {title}" + (f" · from {best.total:.0f} {cur}" if best else "")
    trip_line = f"{r.party_label()} · {r.nights} nights · budget {r.budget_min:.0f}–{r.budget_max:.0f} {cur}"
    notes = [*plan.warnings, "Prices come from Google Flights/Hotels and change often. Check them before booking."]

    # ---- HTML ----
    rows = []
    for i, p in enumerate(plan.packages, start=1):
        f, h = p.flight, p.hotel
        transfer = f" + {p.ground_transfer:.0f} transfer" if p.ground_transfer else ""
        rating = f", {h.rating}★ ({h.reviews or 0} reviews)" if h.rating else ""
        color = _STATUS_COLOR[p.budget_status]
        rows.append(
            "<tr>"
            f'<td style="{_CELL}">{i}</td>'
            f'<td style="{_CELL};color:{color};white-space:nowrap"><b>{p.total:.0f} {cur}</b><br>'
            f"{escape(p.budget_status.value)}</td>"
            f'<td style="{_CELL}">{escape(f.departure_airport)} → {escape(f.arrival_airport)}, {_stops(p)}<br>'
            f"{escape(', '.join(f.airlines))}<br>out {escape(f.outbound_legs[0].departs)}<br>"
            f"{f.price_total:.0f} {cur}{transfer}<br>{_link(f.booking_url, 'Google Flights')}</td>"
            f'<td style="{_CELL}">{escape(h.name)}{escape(rating)}<br>in {escape(h.location_query)}<br>'
            f"{h.total_price:.0f} {cur} total<br>"
            f"{escape(', '.join(h.sellers[:3]))}<br>{_link(h.url, 'Open hotel')}</td>"
            f'<td style="{_CELL}">{escape(p.why)}</td>'
            "</tr>"
        )
    headers = "".join(
        f'<th style="{_CELL};background:#f6f8fa">{c}</th>' for c in ("#", "Total", "Flight", "Stay", "Why")
    )
    packages_html = (
        f'<table style="border-collapse:collapse;font-size:14px"><tr>{headers}</tr>{"".join(rows)}</table>'
        if rows
        else "<p>No flight + hotel packages found.</p>"
    )
    deals_html = "".join(
        f"<li>{_link(d.result.url, d.result.title)}"
        f"{escape(' – ' + d.quoted_price) if d.quoted_price else ''}<br>{escape(d.why)}</li>"
        for d in plan.agency_deals
    )
    u = plan.usage
    html = f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1f2328;max-width:900px">
  <h2 style="margin-bottom:4px">{escape(title)}</h2>
  <p style="margin-top:0;color:#59636e">{escape(trip_line)}</p>
  <p>{escape(plan.summary)}</p>
  <p><b>Departure airport:</b> {escape(plan.origin_note)}</p>
  <h3>Flight + hotel packages</h3>
  {packages_html}
  {f"<h3>Travel agency packages</h3><ul>{deals_html}</ul>" if deals_html else ""}
  <h3>Notes</h3>
  <ul>{"".join(f"<li>{escape(n)}</li>" for n in notes)}</ul>
  <p style="color:#59636e;font-size:12px">{u.llm_calls} LLM calls ({u.total_tokens:,} tokens, {escape(u.llm_model)}),
  {u.serpapi_searches} SerpApi and {u.tavily_searches} Tavily searches.</p>
</div>"""

    # ---- Plain text ----
    lines = [title, trip_line, "", plan.summary, f"Departure airport: {plan.origin_note}", ""]
    for i, p in enumerate(plan.packages, start=1):
        f, h = p.flight, p.hotel
        lines += [
            f"{i}. {p.total:.0f} {cur} ({p.budget_status.value})",
            f"   Flight: {f.departure_airport} -> {f.arrival_airport}, {_stops(p)}, {', '.join(f.airlines)}, "
            f"{f.price_total:.0f} {cur}" + (f" + {p.ground_transfer:.0f} transfer" if p.ground_transfer else ""),
            f"   Stay:   {h.name}, {h.location_query}, {h.total_price:.0f} {cur} total",
            f"   Why:    {p.why}",
        ]
        lines += [f"   {label}: {url}" for label, url in (("Flights", f.booking_url), ("Hotel", h.url)) if url]
    if plan.agency_deals:
        lines += ["", "Travel agency packages:"]
        for d in plan.agency_deals:
            price = f" - {d.quoted_price}" if d.quoted_price else ""
            lines += [f"- {d.result.title}{price}: {d.result.url}", f"  {d.why}"]
    lines += ["", "Notes:", *(f"- {n}" for n in notes)]
    return EmailContent(subject=subject, html=html, text="\n".join(lines))


def sample_plan() -> TravelPlan:
    """A made-up plan for checking email delivery without calling any search or LLM API."""
    request = TripRequest(
        destinations=["Gran Canaria"],
        depart=date(2027, 2, 14),
        return_date=date(2027, 2, 21),
        budget_min=2000,
        budget_max=3500,
        children_ages=[6],
    )
    leg = FlightLeg(
        from_airport="TMP",
        to_airport="LPA",
        departs="2027-02-14 06:10",
        arrives="2027-02-14 13:05",
        airline="Example Air",
    )
    flight = FlightOption(
        id="F1",
        price_total=1450,
        currency="EUR",
        departure_airport="TMP",
        arrival_airport="LPA",
        airlines=["Example Air"],
        stops=0,
        total_duration_min=415,
        outbound_legs=[leg],
        booking_url=None,
    )
    hotel = HotelOption(
        id="H1",
        name="Sample Family Resort",
        location_query="Maspalomas, Gran Canaria",
        kind="hotel",
        total_price=1050,
        price_per_night=150,
        currency="EUR",
        rating=4.4,
        reviews=1200,
        stars=4,
        url=None,
    )
    package = PricedPackage(
        flight=flight,
        hotel=hotel,
        ground_transfer=0,
        total=2500,
        budget_status=BudgetStatus.WITHIN,
        why="Sample row: this is test data, not a real offer.",
    )
    return TravelPlan(
        request=request,
        summary="This is a test email from travel-plan --test-email. The data below is made up.",
        origin_note="Sample only.",
        packages=[package],
        agency_deals=[],
        warnings=["Test email: no real search was made."],
        usage=Usage(llm_model="none (test email)"),
    )
