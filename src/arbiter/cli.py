"""Command line interface.

    arbiter plan      what will this cost before I commit to it
    arbiter gate      run the comparison and set an exit code
    arbiter diff      explain one flagged task
    arbiter history   what previous runs of this suite decided
    arbiter simulate  check the error rates on an agent with known truth
    arbiter init      write a starter suite file
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .config import SuiteConfig, load_suite
from .diff import diff_task
from .gate.decide import run_gate
from .report import (
    render_console,
    render_diff_console,
    render_diff_markdown,
    render_junit,
    render_markdown,
    render_paired_plan_console,
    render_plan_console,
)
from .stats import paired_plan, plan_suite
from .store import Store

app = typer.Typer(
    name="arbiter",
    help="A statistical CI gate for noisy agent evals.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

_EXAMPLE_SUITE = """\
# arbiter suite definition.
#
# The two targets are the builds being compared. They receive the same task and
# the same seed, which is what makes the comparison paired: whatever randomness
# your agent can control is held fixed across both, so the only thing that
# varies is the change under test.
name: my-agent

tasks:
  - id: refund_flow
    input: {baseline_rate: 0.9, candidate_rate: 0.9, coupling: 0.7}
  - id: multi_hop_lookup
    input: {baseline_rate: 0.8, candidate_rate: 0.8, coupling: 0.7}
  # This one is deliberately broken, so the demo has something to find.
  - id: tool_error_recovery
    input: {baseline_rate: 0.85, candidate_rate: 0.6, coupling: 0.7}

baseline:
  kind: python
  ref: arbiter.sim.agent:baseline
  concurrency: 16

candidate:
  kind: python
  ref: arbiter.sim.agent:candidate
  concurrency: 16

stats:
  alpha: 0.05          # false discovery rate across the suite
  beta: 0.10           # miss rate for a regression of the size below
  mde: 0.15            # smallest pass-rate drop worth catching
  odds_ratio: 3.0      # flag when new failures outnumber fixes 3:1
  min_replicates: 4
  max_replicates: 200
  correction: e-bh

budget:
  batch_size: 32
  # max_cost_usd: 25.0
  # max_seconds: 900

gate:
  allocator: cheapest-to-close
  on_inconclusive: warn
"""


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"arbiter {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """A statistical CI gate for noisy agent evals."""


def _load(path: Path) -> SuiteConfig:
    try:
        return load_suite(path)
    except Exception as exc:
        err_console.print(f"[red]could not load {path}:[/red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the suite file.")] = Path(
        "arbiter.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a starter suite file wired to the built-in simulated agent."""
    if path.exists() and not force:
        err_console.print(f"[red]{path} already exists[/red] (pass --force to overwrite)")
        raise typer.Exit(code=1)
    path.write_text(_EXAMPLE_SUITE, encoding="utf-8")
    console.print(f"wrote {path}")
    console.print("run [bold]arbiter gate " + str(path) + "[/bold] to try it")


@app.command()
def plan(
    tasks: Annotated[int, typer.Option("--tasks", "-n", help="Tasks in the suite.")] = 100,
    baseline_rate: Annotated[
        float, typer.Option("--baseline-rate", help="Pass rate you see today.")
    ] = 0.9,
    mde: Annotated[
        float, typer.Option("--mde", help="Smallest pass-rate drop worth catching.")
    ] = 0.15,
    alpha: Annotated[float, typer.Option("--alpha")] = 0.05,
    beta: Annotated[float, typer.Option("--beta")] = 0.10,
    cost: Annotated[
        float, typer.Option("--cost", help="Dollars per replicate.")
    ] = 0.0,
    seconds: Annotated[
        float, typer.Option("--seconds", help="Seconds per replicate.")
    ] = 0.0,
    concurrency: Annotated[int, typer.Option("--concurrency")] = 8,
    coupling: Annotated[
        float,
        typer.Option(
            "--coupling",
            help="Share of replicates where seeding makes the two builds agree.",
        ),
    ] = 0.7,
    odds_ratio: Annotated[float, typer.Option("--odds-ratio")] = 3.0,
    correction: Annotated[str, typer.Option("--correction")] = "e-bh",
) -> None:
    """Estimate what gating a suite of this shape will cost."""
    suite_plan = plan_suite(
        n_tasks=tasks,
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        beta=beta,
        cost_per_replicate=cost,
        seconds_per_replicate=seconds,
        concurrency=concurrency,
    )
    render_plan_console(suite_plan, console)
    render_paired_plan_console(
        paired_plan(
            n_tasks=tasks,
            baseline_rate=baseline_rate,
            mde=mde,
            coupling=coupling,
            alpha=alpha,
            beta=beta,
            odds_ratio=odds_ratio,
            correction=correction,
        ),
        console,
    )


@app.command()
def gate(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML.")],
    store_path: Annotated[
        Path | None,
        typer.Option("--store", help="Run store. Defaults to the suite's own setting."),
    ] = None,
    no_store: Annotated[
        bool,
        typer.Option("--no-store", help="Do not read or write cached runs."),
    ] = False,
    markdown_out: Annotated[
        Path | None, typer.Option("--markdown", help="Write a Markdown report here.")
    ] = None,
    junit_out: Annotated[
        Path | None, typer.Option("--junit", help="Write JUnit XML here.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full result as JSON here.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show every task.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only set the exit code.")] = False,
) -> None:
    """Compare the candidate against the baseline and set an exit code.

    Exit codes: 0 passed, 1 regressions found or the run was rejected by policy,
    2 the gate could not run reliably.
    """
    cfg = _load(suite)
    store = None if no_store else Store(store_path or cfg.store)

    def on_progress(state: dict[str, object]) -> None:
        if quiet:
            return
        console.print(
            f"[dim]{state['resolved']}/{state['total']} tasks decided · "
            f"{state['replicates_run']} replicates · {state['elapsed']:.0f}s[/dim]",
            end="\r",
        )

    try:
        result = asyncio.run(run_gate(cfg, store=store, progress=on_progress))
        if store is not None:
            store.record_gate(result.to_dict())
    finally:
        if store is not None:
            store.close()

    if not quiet:
        console.print(" " * 80, end="\r")
        render_console(result, console, verbose=verbose)

    if markdown_out:
        markdown_out.write_text(render_markdown(result), encoding="utf-8")
    if junit_out:
        junit_out.write_text(render_junit(result), encoding="utf-8")
    if json_out:
        json_out.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    raise typer.Exit(code=result.exit_code)


@app.command()
def diff(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML.")],
    task_id: Annotated[str, typer.Argument(help="Task to explain.")],
    store_path: Annotated[Path | None, typer.Option("--store")] = None,
    markdown: Annotated[
        bool, typer.Option("--markdown", help="Emit Markdown instead of a table.")
    ] = False,
) -> None:
    """Show how a task's behaviour changed, using runs already in the store."""
    cfg = _load(suite)
    store = Store(store_path or cfg.store)
    try:
        baseline_runs = store.load_task_runs(cfg.variant_id("baseline"), task_id)
        candidate_runs = store.load_task_runs(cfg.variant_id("candidate"), task_id)
    finally:
        store.close()

    if not baseline_runs and not candidate_runs:
        err_console.print(
            f"[red]no stored runs for '{task_id}'[/red] — run [bold]arbiter gate[/bold] first, "
            f"and check the task id"
        )
        raise typer.Exit(code=1)

    task_diff = diff_task(task_id, baseline_runs, candidate_runs)
    if markdown:
        sys.stdout.write(render_diff_markdown(task_diff))
    else:
        render_diff_console(task_diff, console)


@app.command()
def history(
    suite: Annotated[Path, typer.Argument(help="Path to the suite YAML.")],
    store_path: Annotated[Path | None, typer.Option("--store")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """List recent gate runs for this suite."""
    cfg = _load(suite)
    store = Store(store_path or cfg.store)
    try:
        rows = store.recent_gates(cfg.name, limit=limit)
    finally:
        store.close()
    if not rows:
        console.print("no gate runs recorded yet")
        return
    from rich.table import Table

    table = Table(header_style="bold")
    table.add_column("when")
    table.add_column("verdict")
    table.add_column("flagged", justify="right")
    table.add_column("tasks", justify="right")
    table.add_column("replicates", justify="right")
    table.add_column("reused", justify="right")
    table.add_column("cost", justify="right")
    import datetime as _dt

    for row in rows:
        when = _dt.datetime.fromtimestamp(row["created_at"]).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            when,
            row["verdict"],
            str(row["n_flagged"]),
            str(row["n_tasks"]),
            str(row["replicates_run"]),
            str(row["replicates_reused"]),
            f"${row['cost_usd']:,.2f}",
        )
    console.print(table)


@app.command()
def simulate(
    trials: Annotated[int, typer.Option("--trials", "-t")] = 40,
    tasks: Annotated[int, typer.Option("--tasks", "-n")] = 25,
    regressed: Annotated[
        int, typer.Option("--regressed", help="How many tasks genuinely got worse.")
    ] = 3,
    coupling: Annotated[
        float, typer.Option("--coupling", help="How much seeding couples the two builds.")
    ] = 0.7,
    correction: Annotated[str, typer.Option("--correction")] = "e-bh",
    allocator: Annotated[str, typer.Option("--allocator")] = "cheapest-to-close",
    max_replicates: Annotated[int, typer.Option("--max-replicates")] = 150,
    json_out: Annotated[Path | None, typer.Option("--json")] = None,
) -> None:
    """Run the gate against an agent whose truth is known, and report the error rates.

    Set --regressed 0 to measure how often the gate flags something on a build
    where nothing changed. That number is the one worth checking before you
    trust any of this.
    """
    from .sim.harness import ScenarioSpec, run_experiment

    spec = ScenarioSpec(
        n_tasks=tasks,
        n_regressed=regressed,
        coupling=coupling,
        correction=correction,
        allocator=allocator,
        max_replicates=max_replicates,
    )
    with console.status(f"running {trials} trials against the simulated agent..."):
        experiment = asyncio.run(run_experiment(spec, n_trials=trials, salt_prefix="cli"))

    from rich.table import Table

    table = Table(
        title=f"{trials} trials · {tasks} tasks · {regressed} genuinely regressed",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("measure")
    table.add_column("value", justify="right")
    table.add_row("false discovery rate", f"{experiment.fdr:.3f}  (bound {spec.alpha})")
    table.add_row("trials with any false flag", f"{experiment.any_false_flag_rate:.3f}")
    if regressed:
        table.add_row("regressed tasks caught", f"{experiment.per_task_power:.3f}")
        table.add_row("builds blocked", f"{experiment.detection_rate:.3f}")
    table.add_row("mean replicates", f"{experiment.mean_replicates:,.0f}")
    table.add_row("fixed-sample equivalent", f"{experiment.fixed_replicates:,}")
    table.add_row("saving", f"{experiment.savings:+.1%}")
    console.print(table)

    if json_out:
        json_out.write_text(json.dumps(experiment.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    app()
