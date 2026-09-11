"""The gate itself: run replicates until the suite has an answer, then answer.

The loop is small on purpose. Ask the allocator which tasks deserve the next
batch, run those replicates against both builds concurrently, feed the pairs to
their tests, charge the budget, repeat until nothing is left undecided or the
money runs out. Everything interesting is in the pieces it calls.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import SuiteConfig
from ..runner.engine import Cell, Engine
from ..runner.types import RunOutcome, Task, seed_for
from ..scheduler.allocator import build_allocator
from ..scheduler.budget import BudgetTracker
from ..stats import Verdict, benjamini_hochberg, e_bh, holm
from ..store import Store
from .task_test import StopReason, TaskResult, TaskTest

__all__ = ["GateResult", "evidence_threshold", "gate_sync", "run_gate"]

ProgressFn = Callable[[dict[str, Any]], None]

# Enough attempts to tell a broken target from an unlucky start.
_MIN_ATTEMPTS_BEFORE_ABORT = 40


def evidence_threshold(n_tasks: int, alpha: float, correction: str) -> float:
    """The e-value a single task must reach to be flagged on its own.

    With no correction that is ``1 / alpha``, straight from Ville's inequality.
    With any of the suite-level procedures the strictest rung is the first one,
    ``n_tasks / alpha``: a lone regression in a big suite has to clear a high
    bar, which is the multiplicity price and is not avoidable. The step-up
    structure means that once several tasks look bad, the bar for each of them
    drops, so a real across-the-board regression is caught far more cheaply than
    this number suggests.
    """
    if correction == "none":
        return 1.0 / alpha
    return n_tasks / alpha


@dataclass
class GateResult:
    """The verdict, the evidence behind it, and what it cost."""

    suite: str
    verdict: str
    tasks: list[TaskResult]
    baseline_variant: str
    candidate_variant: str
    correction: str
    alpha: float
    evidence_threshold: float
    replicates_run: int = 0
    replicates_reused: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    infra_errors: int = 0
    infra_error_rate: float = 0.0
    infra_error_examples: list[str] = field(default_factory=list)
    binding_constraint: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[TaskResult]:
        return [t for t in self.tasks if t.flagged]

    @property
    def inconclusive(self) -> list[TaskResult]:
        return [t for t in self.tasks if t.verdict is Verdict.INCONCLUSIVE]

    @property
    def cleared(self) -> list[TaskResult]:
        return [t for t in self.tasks if t.verdict is Verdict.PASS]

    @property
    def no_evidence(self) -> list[TaskResult]:
        """Tasks that got their full allotment and showed nothing.

        Worth separating from :attr:`incomplete`. "I looked as hard as I said I
        would and found nothing" and "I ran out of money halfway" are very
        different statements, and collapsing them into one bucket called
        inconclusive is how a gate ends up either crying wolf on every clean
        build or going quiet on a truncated one.
        """
        return [
            t
            for t in self.tasks
            if t.verdict is Verdict.INCONCLUSIVE
            and t.stop_reason in (StopReason.FUTILITY, StopReason.MAX_REPLICATES)
        ]

    @property
    def incomplete(self) -> list[TaskResult]:
        """Tasks the suite budget cut short before they could decide."""
        return [t for t in self.tasks if t.stop_reason is StopReason.BUDGET]

    @property
    def exit_code(self) -> int:
        return {"pass": 0, "warn": 0, "fail": 1, "error": 2}.get(self.verdict, 1)

    @property
    def replicates_total(self) -> int:
        return self.replicates_run + self.replicates_reused

    @property
    def reuse_rate(self) -> float:
        total = self.replicates_total
        return self.replicates_reused / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "verdict": self.verdict,
            "baseline_variant": self.baseline_variant,
            "candidate_variant": self.candidate_variant,
            "correction": self.correction,
            "alpha": self.alpha,
            "evidence_threshold": self.evidence_threshold,
            "n_tasks": len(self.tasks),
            "n_flagged": len(self.flagged),
            "n_cleared": len(self.cleared),
            "n_inconclusive": len(self.inconclusive),
            "n_no_evidence": len(self.no_evidence),
            "n_incomplete": len(self.incomplete),
            "replicates_run": self.replicates_run,
            "replicates_reused": self.replicates_reused,
            "cost_usd": self.cost_usd,
            "wall_seconds": self.wall_seconds,
            "infra_errors": self.infra_errors,
            "infra_error_rate": self.infra_error_rate,
            "infra_error_examples": self.infra_error_examples,
            "binding_constraint": self.binding_constraint,
            "notes": self.notes,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def _apply_correction(tests: list[TaskTest], alpha: float, correction: str) -> tuple[
    set[int], list[float]
]:
    """Decide which tasks survive suite-level correction."""
    if not tests:
        return set(), []
    e_values = [t.e_value for t in tests]
    p_values = [t.anytime_p for t in tests]
    if correction == "e-bh":
        result = e_bh(e_values, alpha)
    elif correction == "bh":
        result = benjamini_hochberg(p_values, alpha)
    elif correction == "holm":
        result = holm(p_values, alpha)
    else:
        rejected = [i for i, p in enumerate(p_values) if p <= alpha]
        return set(rejected), p_values
    return set(result.rejected), result.adjusted


async def run_gate(
    cfg: SuiteConfig,
    *,
    store: Store | None = None,
    progress: ProgressFn | None = None,
) -> GateResult:
    """Run the suite and return a verdict.

    ``store`` is optional but strongly recommended: without it every gate run
    re-executes the baseline from scratch, which roughly doubles the bill for no
    statistical benefit.
    """
    started = time.monotonic()
    tasks: list[Task] = cfg.to_tasks()
    by_id = {t.id: t for t in tasks}
    baseline_variant = cfg.variant_id("baseline")
    candidate_variant = cfg.variant_id("candidate")

    threshold = evidence_threshold(len(tasks), cfg.stats.alpha, cfg.stats.correction)
    # Translate the flagging threshold into the alpha the per-task stopping
    # boundary should use, so that a task which stops has enough evidence to
    # survive correction on its own.
    alpha_effective = min(max((1.0 - cfg.stats.beta) / threshold, 1e-12), cfg.stats.alpha)

    tests = {
        t.id: TaskTest(
            task=t,
            stats=cfg.stats,
            alpha_effective=alpha_effective,
            futility_confidence=cfg.stats.futility_confidence,
        )
        for t in tasks
    }
    allocator = build_allocator(cfg.gate.allocator, cfg.stats.min_replicates)
    budget = BudgetTracker(cfg.budget)

    cached_baseline: dict[tuple[str, int], RunOutcome] = {}
    cached_candidate: dict[tuple[str, int], RunOutcome] = {}
    if store is not None:
        ids = [t.id for t in tasks]
        cached_baseline = store.load_runs(baseline_variant, ids)
        cached_candidate = store.load_runs(candidate_variant, ids)

    baseline_engine = Engine(cfg.baseline)
    candidate_engine = Engine(cfg.candidate)
    next_replicate = dict.fromkeys(by_id, 0)
    replicates_run = 0
    replicates_reused = 0
    notes: list[str] = []

    try:
        while True:
            for task_id, test in tests.items():
                if next_replicate[task_id] >= cfg.stats.max_replicates:
                    test.mark_capped()
            pending = [t for t in tests.values() if not t.resolved]
            if not pending:
                break
            if budget.exhausted:
                for test in pending:
                    test.mark_budget_exhausted()
                break

            headroom = budget.headroom()
            if headroom <= 0:
                for test in pending:
                    test.mark_budget_exhausted()
                break

            picks = allocator.select(list(tests.values()), headroom)
            assignments: list[tuple[str, int]] = []
            for task_id in picks:
                replicate = next_replicate[task_id]
                if replicate >= cfg.stats.max_replicates:
                    continue
                next_replicate[task_id] = replicate + 1
                assignments.append((task_id, replicate))
            if not assignments:
                continue

            baseline_cells: list[Cell] = []
            candidate_cells: list[Cell] = []
            pairs: dict[tuple[str, int], list[RunOutcome | None]] = {}
            for task_id, replicate in assignments:
                seed = seed_for(task_id, replicate, cfg.seed_salt)
                cell = Cell(task=by_id[task_id], replicate=replicate, seed=seed)
                cached_b = cached_baseline.get((task_id, replicate))
                cached_c = cached_candidate.get((task_id, replicate))
                pairs[(task_id, replicate)] = [cached_b, cached_c]
                if cached_b is None:
                    baseline_cells.append(cell)
                else:
                    replicates_reused += 1
                if cached_c is None:
                    candidate_cells.append(cell)
                else:
                    replicates_reused += 1

            baseline_results, candidate_results = await asyncio.gather(
                baseline_engine.run_cells(baseline_cells),
                candidate_engine.run_cells(candidate_cells),
            )
            for result in baseline_results:
                pairs[result.cell.key][0] = result.outcome
            for result in candidate_results:
                pairs[result.cell.key][1] = result.outcome

            # The budget is charged for every attempt, because a call that
            # timed out still cost money and time. What gets *reported* as
            # replicates run is the number that produced a usable result: a run
            # that reported 1440 attempts and 1440 failures as a 50% error rate
            # would be understating the problem by half.
            attempted = len(baseline_cells) + len(candidate_cells)
            succeeded = sum(1 for r in baseline_results if r.ok) + sum(
                1 for r in candidate_results if r.ok
            )
            replicates_run += succeeded
            batch_cost = baseline_engine.stats.cost_usd + candidate_engine.stats.cost_usd
            budget.charge(replicates=attempted)
            budget.cost_usd = batch_cost

            for (task_id, replicate), (base_outcome, cand_outcome) in pairs.items():
                test = tests[task_id]
                if base_outcome is None or cand_outcome is None:
                    # One half of the pair never ran, so there is no comparison
                    # to make. Counted as harness trouble, not as evidence.
                    test.record_infra_error()
                    continue
                test.observe(cand_outcome, base_outcome)
                if store is None:
                    continue
                # Only write back what was actually executed. The insert is
                # idempotent either way, but re-gating a fully cached suite would
                # otherwise issue thousands of statements that change nothing.
                seed = seed_for(task_id, replicate, cfg.seed_salt)
                if (task_id, replicate) not in cached_baseline:
                    store.record_run(
                        variant_id=baseline_variant,
                        suite=cfg.name,
                        task_id=task_id,
                        replicate=replicate,
                        seed=seed,
                        outcome=base_outcome,
                    )
                if (task_id, replicate) not in cached_candidate:
                    store.record_run(
                        variant_id=candidate_variant,
                        suite=cfg.name,
                        task_id=task_id,
                        replicate=replicate,
                        seed=seed,
                        outcome=cand_outcome,
                    )
            if store is not None:
                store.commit()

            # Stop early when the harness is clearly broken. Grinding through the
            # whole budget against a target that fails every call wastes real
            # minutes and cannot change the verdict.
            errors_so_far = (
                baseline_engine.stats.infra_errors + candidate_engine.stats.infra_errors
            )
            attempts_so_far = replicates_run + errors_so_far
            if (
                attempts_so_far >= _MIN_ATTEMPTS_BEFORE_ABORT
                and errors_so_far / attempts_so_far > cfg.gate.max_infra_error_rate
            ):
                notes.append(
                    f"abandoned the run after {attempts_so_far} attempts: "
                    f"{errors_so_far} of them could not be executed"
                )
                for test in tests.values():
                    test.mark_budget_exhausted()
                break

            if progress is not None:
                progress(
                    {
                        "resolved": sum(1 for t in tests.values() if t.resolved),
                        "total": len(tests),
                        "replicates_run": replicates_run,
                        "replicates_reused": replicates_reused,
                        "cost_usd": budget.cost_usd,
                        "elapsed": budget.elapsed_seconds,
                    }
                )
    finally:
        await baseline_engine.aclose()
        await candidate_engine.aclose()

    ordered = list(tests.values())
    rejected, adjusted = _apply_correction(ordered, cfg.stats.alpha, cfg.stats.correction)

    results: list[TaskResult] = []
    for index, test in enumerate(ordered):
        task_result = test.result()
        task_result.adjusted_p = adjusted[index] if index < len(adjusted) else 1.0
        task_result.flagged = index in rejected
        if task_result.flagged and task_result.verdict is not Verdict.REGRESSION:
            # e-BH is a step-up procedure, so a task can be flagged on the
            # strength of the company it keeps even though its own sequential
            # test never crossed. That is the procedure working, not a bug.
            task_result.verdict = Verdict.REGRESSION
        results.append(task_result)

    infra_errors = baseline_engine.stats.infra_errors + candidate_engine.stats.infra_errors
    total_attempts = replicates_run + infra_errors
    infra_rate = infra_errors / total_attempts if total_attempts else 0.0
    error_examples = list(
        dict.fromkeys(baseline_engine.stats.errors + candidate_engine.stats.errors)
    )

    n_flagged = sum(1 for r in results if r.flagged)
    # Only a truncated run counts against the inconclusive policy. A task that
    # spent its whole allotment and found nothing has answered the question it
    # was asked.
    n_incomplete = sum(1 for r in results if r.stop_reason is StopReason.BUDGET)

    # Stated regardless of which branch decides the verdict below. A run that
    # both found a regression and ran out of money is two separate facts, and
    # the second one does not stop being true because the first one set the
    # exit code.
    if n_incomplete:
        notes.append(
            f"{n_incomplete} tasks were cut short by the budget before deciding"
        )

    if infra_rate > cfg.gate.max_infra_error_rate:
        verdict = "error"
        notes.append(
            f"{infra_errors} replicates could not be run ({infra_rate:.0%} of attempts), "
            f"above the {cfg.gate.max_infra_error_rate:.0%} ceiling; the verdict is not "
            f"trustworthy"
        )
        # Without an example the reader has to go and reproduce it by hand, which
        # is exactly the situation this note exists to prevent.
        for example in error_examples[:2]:
            notes.append(f"target error: {example}")
    elif n_flagged > cfg.gate.max_flagged_tasks and cfg.gate.fail_on_any_flag:
        verdict = "fail"
    elif n_incomplete and cfg.gate.on_inconclusive == "fail":
        verdict = "fail"
        notes.append("policy is to fail when tasks are cut short")
    elif n_incomplete and cfg.gate.on_inconclusive == "warn":
        verdict = "warn"
    else:
        verdict = "pass"

    if budget.binding_constraint:
        notes.append(f"stopped early on {budget.binding_constraint}")
    notes.extend(_power_diagnostics(ordered, cfg, threshold))

    return GateResult(
        suite=cfg.name,
        verdict=verdict,
        tasks=results,
        baseline_variant=baseline_variant,
        candidate_variant=candidate_variant,
        correction=cfg.stats.correction,
        alpha=cfg.stats.alpha,
        evidence_threshold=threshold,
        replicates_run=replicates_run,
        replicates_reused=replicates_reused,
        cost_usd=baseline_engine.stats.cost_usd + candidate_engine.stats.cost_usd,
        wall_seconds=time.monotonic() - started,
        infra_errors=infra_errors,
        infra_error_rate=infra_rate,
        infra_error_examples=error_examples[:5],
        binding_constraint=budget.binding_constraint,
        notes=notes,
    )


def _power_diagnostics(
    tests: list[TaskTest], cfg: SuiteConfig, threshold: float
) -> list[str]:
    """Tell the user when the run was never capable of finding anything.

    A gate that cannot reach its own evidence threshold inside its replicate cap
    is not a passing gate, it is a broken one, and it will go green forever
    while regressions ship. This is the check that says so out loud, using the
    disagreement rate actually observed rather than an assumed one.

    Note that the arithmetic uses the drift under the *alternative*, not the
    boundary distance divided by a single regression's step. Disagreements do
    not all point the same way even when the candidate really is worse, and a
    diagnostic that assumes they do will cheerfully report an underpowered suite
    as adequate, which is the one thing it exists to prevent.
    """
    if not tests or cfg.stats.mode != "binary":
        return []
    spec = tests[0].paired.inner.spec
    drift = spec.drift(spec.p1)
    if drift <= 0:
        return []
    pairs_needed = spec.upper / drift
    observed_discordant = sum(t.paired.discordant for t in tests)
    observed_replicates = sum(t.replicates for t in tests)
    if not observed_replicates:
        return []
    rate = observed_discordant / observed_replicates
    notes = []
    if rate <= 0:
        notes.append(
            "the two builds never disagreed on any replicate; either they are identical "
            "or the target is ignoring the seed it is given"
        )
        return notes
    replicates_needed = pairs_needed / rate
    if replicates_needed > cfg.stats.max_replicates:
        notes.append(
            f"underpowered: flagging one task needs about {pairs_needed:.0f} disagreeing "
            f"replicates, the builds disagreed on {rate:.1%} of runs, so about "
            f"{replicates_needed:.0f} replicates per task. max_replicates is "
            f"{cfg.stats.max_replicates}. Raise it, raise alpha, or lower odds_ratio."
        )
    return notes


def gate_sync(cfg: SuiteConfig, **kwargs: Any) -> GateResult:
    """Blocking wrapper, for callers that are not already in an event loop."""
    return asyncio.run(run_gate(cfg, **kwargs))
