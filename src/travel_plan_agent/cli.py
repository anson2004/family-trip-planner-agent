from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.table import Table

from travel_plan_agent.models import BudgetStatus, TravelPlan

if TYPE_CHECKING:
    from travel_plan_agent.config import Settings

app = typer.Typer(add_completion=False, help="Plan a holiday from Finland: flights + hotel within a budget.")
console = Console()

_STATUS_STYLE = {BudgetStatus.WITHIN: "green", BudgetStatus.UNDER: "cyan", BudgetStatus.OVER: "red"}


def _load_settings[T: BaseSettings](cls: type[T], env_file: Path) -> T:
    try:
        return cls(_env_file=env_file)
    except ValidationError as exc:
        problems = "\n".join(f"  {str(e['loc'][0]).upper()}: {e['msg']}" for e in exc.errors())
        console.print(f"[red]Fix these settings in {env_file} (see .env.example):\n{problems}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def plan(
    env_file: Annotated[Path, typer.Option("--env-file", help="Settings file with keys and TRIP_* values")] = Path(
        ".env"
    ),
    json_out: Annotated[Path | None, typer.Option("--json", help="Also save the plan as JSON to this file")] = None,
    quiet: Annotated[bool, typer.Option(help="Hide search progress")] = False,
    trace: Annotated[bool, typer.Option(help="Show every LLM call with its token count")] = False,
    no_email: Annotated[bool, typer.Option("--no-email", help="Don't email the plan, even if EMAIL_TO is set")] = False,
    test_email: Annotated[
        bool, typer.Option("--test-email", help="Only send a sample email to check Resend; no searches or LLM calls")
    ] = False,
) -> None:
    """Plan the trip described by the TRIP_* settings in the env file."""
    if not env_file.exists():
        console.print(f"[red]{env_file} not found. Copy .env.example to .env and fill it in.[/red]")
        raise typer.Exit(1)

    # Imported here so `--help` stays fast.
    from travel_plan_agent.agent import plan_trip
    from travel_plan_agent.config import Settings, TripSettings

    settings = _load_settings(Settings, env_file)
    if test_email:
        if not settings.email_to or settings.resend_api_key is None:
            console.print(f"[red]Set RESEND_API_KEY and EMAIL_TO in {env_file} first.[/red]")
            raise typer.Exit(1)
        from travel_plan_agent.email_report import sample_plan

        _email(sample_plan(), settings, subject_prefix="[TEST] ")
        return
    trip = _load_settings(TripSettings, env_file)
    try:
        request = trip.to_request()
    except ValidationError as exc:
        problems = "; ".join(e["msg"] for e in exc.errors())
        console.print(f"[red]Invalid trip settings in {env_file}: {problems}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        f"Planning [bold]{request.origin_label} -> {request.destination_label}[/bold], {request.dates_label()}, "
        f"{request.party_label()}, "
        f"budget {request.budget_min:.0f}-{request.budget_max:.0f} {request.currency}"
    )
    log = (lambda _m: None) if quiet else (lambda m: console.print(f"  [dim]searching: {m}[/dim]"))
    with console.status("Agent is working..."):
        result = plan_trip(request, settings, log=log)

    render(result)
    if trace:
        render_trace(result)
    if json_out:
        json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Saved to {json_out}")
    if not no_email:
        _email(result, settings)


def _email(result: TravelPlan, settings: "Settings", subject_prefix: str = "") -> None:
    if not settings.email_to:
        return
    if settings.resend_api_key is None:
        console.print("[yellow]EMAIL_TO is set but RESEND_API_KEY is empty, so no email was sent.[/yellow]")
        return
    from travel_plan_agent.email_report import build_email
    from travel_plan_agent.providers.resend import ResendError, send_email

    content = build_email(result)
    content = replace(content, subject=subject_prefix + content.subject)
    try:
        send_email(settings.resend_api_key.get_secret_value(), settings.email_from, settings.email_to, content)
    except (ResendError, httpx.HTTPError) as exc:
        console.print(f"[red]Email not sent: {exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Emailed the plan to {', '.join(settings.email_to)}")


def render(plan: TravelPlan) -> None:
    cur = plan.request.currency
    console.rule("Travel plan")
    console.print(plan.summary)
    console.print(f"[bold]Departure airport:[/bold] {plan.origin_note}\n")

    if plan.packages:
        table = Table(show_lines=True)
        for col in ("#", "Total", "Flight", "Stay", "Why"):
            table.add_column(col)
        for i, p in enumerate(plan.packages, start=1):
            f, h = p.flight, p.hotel
            out = f.outbound_legs
            flight = (
                f"{f.departure_airport}->{f.arrival_airport}, {'nonstop' if f.stops == 0 else f'{f.stops} stop(s)'}\n"
                f"{f.outbound_date:%a %d %b} – {f.return_date:%a %d %b}\n"
                f"{', '.join(f.airlines)}\nout {out[0].departs}\n{f.price_total:.0f} {cur}"
            )
            if p.ground_transfer:
                flight += f" + {p.ground_transfer:.0f} transfer"
            if f.booking_url:
                flight += f"\n[link={f.booking_url}]Google Flights[/link]"
            rating = f", {h.rating}★ ({h.reviews or 0})" if h.rating else ""
            stay = f"{h.name}{rating}\nin {h.location_query}\n{h.total_price:.0f} {cur} total"
            if h.sellers:
                stay += f"\nvia {', '.join(h.sellers[:3])}"
            if h.url:
                stay += f"\n[link={h.url}]Open[/link]"
            style = _STATUS_STYLE[p.budget_status]
            total = f"[{style}]{p.total:.0f} {cur}\n{p.budget_status.value}[/{style}]"
            table.add_row(str(i), total, flight, stay, p.why)
        console.print(table)
    else:
        console.print("[yellow]No flight + hotel packages found.[/yellow]")

    if plan.agency_deals:
        console.print("\n[bold]Travel agency packages[/bold]")
        for d in plan.agency_deals:
            price = f" - {d.quoted_price}" if d.quoted_price else ""
            console.print(f"• [link={d.result.url}]{d.result.title}[/link]{price}\n  {d.why}\n  {d.result.url}")

    warnings = [*plan.warnings, "Prices come from Google Flights/Hotels and change often. Check them before booking."]
    console.print("\n[bold]Notes[/bold]")
    for w in warnings:
        console.print(f"• {w}")
    u = plan.usage
    console.print(
        f"\n[bold]Usage[/bold] ({u.llm_model}): {u.llm_calls} LLM calls, "
        f"{u.input_tokens:,} input + {u.output_tokens:,} output = {u.total_tokens:,} tokens\n"
        f"Live searches: SerpApi {u.serpapi_searches}, Tavily {u.tavily_searches} (cached searches are free)"
    )


def render_trace(plan: TravelPlan) -> None:
    table = Table(title="LLM calls")
    for col, justify in (("#", "right"), ("What the LLM decided", "left"), ("Input", "right"), ("Output", "right")):
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for i, c in enumerate(plan.usage.calls, start=1):
        table.add_row(str(i), c.action, f"{c.input_tokens:,}", f"{c.output_tokens:,}")
    u = plan.usage
    table.add_row("", "[bold]Total[/bold]", f"[bold]{u.input_tokens:,}[/bold]", f"[bold]{u.output_tokens:,}[/bold]")
    console.print(table)
