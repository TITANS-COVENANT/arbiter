"""Terminal rendering.

The console report leads with the verdict and the two or three tasks that
caused it. Everything else is available behind a flag. A gate that prints two
hundred rows every run is a gate nobody reads.
"""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..diff import TaskDiff
from ..gate.decide import GateResult
from ..stats import PairedPlan, SuitePlan

__all__ = [
    "render_console",
    "render_diff_console",
    "render_paired_plan_console",
    "render_plan_console",
]

_STYLE = {"pass": "green", "warn": "yellow", "fail": "red", "error": "magenta"}
_HEADLINE = {
    "pass": "No regressions found",
    "warn": "No regressions found, but the run was cut short",
    "fail": "Regressions found",
    "error": "The gate could not run reliably",
}


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_console(result: GateResult, console: Console, *, verbose: bool = False) -> None:
    """Print a gate result."""
    style = _STYLE[result.verdict]
    console.print(
        Panel(
            Text(
                f"{_HEADLINE[result.verdict]}\n"
                f"{len(result.flagged)} flagged / {len(result.tasks)} tasks  ·  "
                f"{result.replicates_run} replicates run"
                + (f", {result.replicates_reused} reused" if result.replicates_reused else "")
                + (f"  ·  ${result.cost_usd:,.2f}" if result.cost_usd else "")
                + f"  ·  {result.wall_seconds:.0f}s",
                style=style,
            ),
            title=f"arbiter · {result.suite}",
            border_style=style,
        )
    )

    flagged = result.flagged
    if flagged:
        table = Table(title="Flagged as regressed", title_justify="left", header_style="bold")
        table.add_column("task")
        table.add_column("baseline", justify="right")
        table.add_column("candidate", justify="right")
        table.add_column("broke", justify="right")
        table.add_column("fixed", justify="right")
        table.add_column("reps", justify="right")
        table.add_column("e-value", justify="right")
        table.add_column("adj p", justify="right")
        for task in sorted(flagged, key=lambda t: t.adjusted_p):
            table.add_row(
                task.task_id,
                _pct(task.baseline_rate),
                _pct(task.candidate_rate),
                str(task.regressions),
                str(task.improvements),
                str(task.replicates),
                f"{task.e_value:,.0f}",
                f"{task.adjusted_p:.4f}",
            )
        console.print(table)

    counts = Table.grid(padding=(0, 2))
    counts.add_row("cleared outright", str(len(result.cleared)))
    counts.add_row("no evidence of a regression", str(len(result.no_evidence)))
    if result.incomplete:
        counts.add_row("cut short by the budget", str(len(result.incomplete)))
    if result.infra_errors:
        counts.add_row(
            "replicates that could not run",
            f"{result.infra_errors} ({_pct(result.infra_error_rate)})",
        )
    console.print(counts)

    for note in result.notes:
        console.print(f"[yellow]note[/yellow] {note}")

    for example in result.infra_error_examples[:3]:
        console.print(f"[magenta]target error[/magenta] {example}")

    if verbose:
        detail = Table(title="All tasks", title_justify="left", header_style="bold")
        detail.add_column("task")
        detail.add_column("verdict")
        detail.add_column("stopped because")
        detail.add_column("reps", justify="right")
        detail.add_column("broke", justify="right")
        detail.add_column("fixed", justify="right")
        detail.add_column("e-value", justify="right")
        for task in result.tasks:
            detail.add_row(
                task.task_id,
                task.verdict.value,
                task.stop_reason.label,
                str(task.replicates),
                str(task.regressions),
                str(task.improvements),
                f"{task.e_value:,.1f}",
            )
        console.print(detail)


def render_diff_console(diff: TaskDiff, console: Console) -> None:
    """Print the explanation for one task."""
    console.print(
        Panel(
            Text(
                f"baseline {diff.baseline_passes}/{diff.baseline_runs} ({_pct(diff.baseline_rate)})"
                f"   candidate {diff.candidate_passes}/{diff.candidate_runs} "
                f"({_pct(diff.candidate_rate)})"
            ),
            title=diff.task_id,
        )
    )

    if diff.modes:
        table = Table(title="Failure modes", title_justify="left", header_style="bold")
        table.add_column("baseline", justify="right")
        table.add_column("candidate", justify="right")
        table.add_column("mode")
        for mode in diff.modes[:10]:
            marker = " [red](new)[/red]" if mode.is_new else ""
            table.add_row(
                str(mode.baseline_count), str(mode.candidate_count), mode.signature + marker
            )
        console.print(table)

    if diff.alignment is not None:
        console.print(
            f"[bold]Replicate {diff.representative_replicate}[/bold] "
            f"(baseline passed, candidate failed, same seed)"
        )
        for line in diff.alignment.summary():
            colour = "red" if line.startswith(("~", "+", "-")) else "dim"
            console.print(f"  [{colour}]{line}[/{colour}]")


def render_plan_console(plan: SuitePlan, console: Console) -> None:
    """Print the unpaired cost comparison, for context only.

    This is the textbook comparison: a fixed-sample proportion test against a
    sequential one, both unpaired and both ignoring multiplicity. It is useful
    for seeing where sequential testing helps at all, and it is *not* the number
    to size a suite from, because arbiter runs a paired test under suite-level
    correction. :func:`render_paired_plan_console` is the one to act on.
    """
    table = Table(
        title="Sequential vs fixed sample, unpaired and uncorrected (for context)",
        title_justify="left",
    )
    table.add_column("")
    table.add_column("fixed sample", justify="right")
    table.add_column("sequential", justify="right")
    table.add_row(
        "replicates per task",
        str(plan.per_task.fixed_samples),
        f"{plan.per_task.expected_samples:.0f}",
    )
    table.add_row(
        "replicates total", str(plan.fixed_replicates), f"{plan.expected_replicates:.0f}"
    )
    if plan.cost_per_replicate:
        table.add_row("cost", f"${plan.fixed_cost:,.2f}", f"${plan.expected_cost:,.2f}")
    if plan.seconds_per_replicate:
        table.add_row(
            "wall clock",
            f"{plan.fixed_wall_clock_seconds / 60:,.0f} min",
            f"{plan.expected_wall_clock_seconds / 60:,.0f} min",
        )
    console.print(table)
    console.print(
        "[dim]Estimates use Wald's approximation and assume nine builds in ten are clean. "
        "They ignore baseline reuse, which saves a further chunk on every run after the "
        "first. Measure the real thing with [/dim][bold]arbiter simulate[/bold]."
    )


def render_paired_plan_console(plan: PairedPlan, console: Console) -> None:
    """Print what to set max_replicates to, and why. This is the actionable table."""
    table = Table(
        title="Sizing the paired test, corrected across the suite (use this)",
        title_justify="left",
    )
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("tasks in the suite", str(plan.n_tasks))
    table.add_row("e-value one task must reach", f"{plan.evidence_threshold:,.0f}")
    table.add_row("disagreeing replicates needed", f"{plan.discordant_pairs_needed:.0f}")
    table.add_row("expected disagreement rate", f"{plan.discordance_rate:.1%}")
    table.add_row("implied odds ratio at that effect", f"{plan.implied_odds_ratio:,.1f}")
    if math.isfinite(plan.replicates_needed):
        table.add_row("[bold]max_replicates[/bold]", f"[bold]{plan.replicates_needed:.0f}[/bold]")
    else:
        table.add_row("[bold]max_replicates[/bold]", "[red]unreachable[/red]")
    console.print(table)
    if not math.isfinite(plan.replicates_needed):
        console.print(
            "[red]No replicate count reaches the boundary:[/red] the odds ratio you asked the "
            "test to detect is larger than the one this effect actually produces. Lower "
            "odds_ratio, or accept a larger mde."
        )
        return
    console.print(
        "[dim]Only replicates where the two builds disagree carry information about the "
        "change. A suite where they agree almost always needs more runs, not fewer.[/dim]"
    )
