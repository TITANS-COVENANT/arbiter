"""Markdown rendering, for pull request comments and CI job summaries.

Written for the person who did not run the gate and does not want to learn what
an e-value is. The verdict is one line at the top, the flagged tasks come next
with the counts that justify the flag, and the statistical machinery is at the
bottom for whoever asks.
"""

from __future__ import annotations

from ..diff import TaskDiff
from ..gate.decide import GateResult

__all__ = ["render_diff_markdown", "render_markdown"]

_HEADLINE = {
    "pass": "No regressions found",
    "warn": "No regressions found, but the run was cut short",
    "fail": "Regressions found",
    "error": "The gate could not run reliably",
}
_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "error": "ERROR"}


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_markdown(result: GateResult, *, title: str = "arbiter") -> str:
    """Render a gate result as a self-contained Markdown report."""
    lines: list[str] = []
    lines.append(f"## {title}: {_ICON[result.verdict]} — {_HEADLINE[result.verdict]}")
    lines.append("")

    flagged = result.flagged
    if flagged:
        lines.append(
            f"{len(flagged)} of {len(result.tasks)} tasks regressed, at a false discovery rate "
            f"of at most {_pct(result.alpha)} across the suite."
        )
    else:
        lines.append(
            f"{len(result.tasks)} tasks compared against the baseline. Nothing crossed the "
            f"evidence threshold."
        )
    lines.append("")

    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Tasks | {len(result.tasks)} |")
    lines.append(f"| Flagged as regressed | {len(flagged)} |")
    lines.append(f"| Cleared outright | {len(result.cleared)} |")
    lines.append(f"| No evidence of a regression | {len(result.no_evidence)} |")
    if result.incomplete:
        lines.append(f"| Cut short by the budget | {len(result.incomplete)} |")
    lines.append(
        f"| Replicates | {result.replicates_run} run"
        + (f", {result.replicates_reused} reused from cache" if result.replicates_reused else "")
        + " |"
    )
    if result.cost_usd:
        lines.append(f"| Cost | ${result.cost_usd:,.2f} |")
    lines.append(f"| Wall clock | {result.wall_seconds:.0f}s |")
    if result.infra_errors:
        lines.append(
            f"| Replicates that could not run | {result.infra_errors} "
            f"({_pct(result.infra_error_rate)}) |"
        )
    lines.append("")

    if flagged:
        lines.append("### Flagged")
        lines.append("")
        lines.append("| Task | Baseline | Candidate | Broke | Fixed | Replicates | Adjusted p |")
        lines.append("|---|---|---|---|---|---|---|")
        for task in sorted(flagged, key=lambda t: t.adjusted_p):
            lines.append(
                f"| `{task.task_id}` | {_pct(task.baseline_rate)} | {_pct(task.candidate_rate)} "
                f"| {task.regressions} | {task.improvements} | {task.replicates} "
                f"| {task.adjusted_p:.4f} |"
            )
        lines.append("")
        lines.append(
            "*Broke* counts replicates where the baseline passed and the candidate failed at "
            "the same seed; *fixed* counts the reverse. Those are the only replicates that "
            "carry information about the change."
        )
        lines.append("")

    incomplete = result.incomplete
    if incomplete:
        lines.append("### Cut short")
        lines.append("")
        lines.append(
            f"{len(incomplete)} tasks stopped before deciding because the run hit its "
            f"`{result.binding_constraint}` ceiling. They are not evidence of anything."
        )
        lines.append("")
        lines.append(", ".join(f"`{t.task_id}`" for t in incomplete[:20]))
        if len(incomplete) > 20:
            lines.append(f" and {len(incomplete) - 20} more")
        lines.append("")

    if result.infra_error_examples:
        lines.append("### Target errors")
        lines.append("")
        lines.append("```")
        lines.extend(result.infra_error_examples[:3])
        lines.append("```")
        lines.append("")

    if result.notes:
        lines.append("### Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in result.notes)
        lines.append("")

    lines.append("<details><summary>How this was decided</summary>")
    lines.append("")
    lines.append(
        "Each task ran under both builds at identical seeds. Replicates where the two "
        "builds disagreed drove a sequential test whose null is that disagreements fall "
        "either way with equal probability. Tasks stopped as soon as that test decided, "
        "or once it could no longer reach significance within "
        "`max_replicates`."
    )
    lines.append("")
    lines.append(
        f"Flagging used `{result.correction}` at alpha={result.alpha}, which for "
        f"{len(result.tasks)} tasks means a single task needs an e-value of "
        f"{result.evidence_threshold:,.0f} to be flagged on its own, and less than that when "
        f"other tasks look bad too."
    )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def render_diff_markdown(diff: TaskDiff) -> str:
    """Render the explanation for a single flagged task."""
    lines = [f"### `{diff.task_id}`", ""]
    lines.append(
        f"Baseline passed {diff.baseline_passes}/{diff.baseline_runs} "
        f"({_pct(diff.baseline_rate)}), candidate passed {diff.candidate_passes}/"
        f"{diff.candidate_runs} ({_pct(diff.candidate_rate)})."
    )
    lines.append("")

    new_modes = diff.new_modes
    if new_modes:
        lines.append("**Failure modes the candidate introduced**")
        lines.append("")
        for mode in new_modes[:5]:
            lines.append(f"- {mode.candidate_count}x `{mode.signature}`")
        lines.append("")

    changed = [m for m in diff.modes if m.delta > 0 and not m.is_new]
    if changed:
        lines.append("**Failure modes that got more common**")
        lines.append("")
        for mode in changed[:5]:
            lines.append(
                f"- `{mode.signature}`: {mode.baseline_count} -> {mode.candidate_count}"
            )
        lines.append("")

    if diff.fixed_modes:
        lines.append("**Failure modes the candidate fixed**")
        lines.append("")
        for mode in diff.fixed_modes[:5]:
            lines.append(f"- {mode.baseline_count}x `{mode.signature}`")
        lines.append("")

    if diff.alignment is not None and diff.representative_replicate is not None:
        lines.append(
            f"**Where replicate {diff.representative_replicate} diverged** "
            f"(baseline passed, candidate failed, same seed)"
        )
        lines.append("")
        lines.append("```")
        lines.extend(diff.alignment.summary())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)
