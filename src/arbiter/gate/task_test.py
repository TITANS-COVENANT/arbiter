"""The per-task decision object.

One of these exists for every task in flight. It swallows paired replicates,
tracks the evidence, and answers three questions the scheduler and the gate
need: has it decided, how suspicious is it, and how much more would it cost to
finish.

A note on what is and is not guaranteed here, because it is the whole point of
the project:

*Flagging* is driven by the paired McNemar sequential test, whose null is an
exact Bernoulli(1/2) on discordant pairs. There is no nuisance parameter to
estimate and no baseline rate to plug in, so the e-value it emits is valid, and
the suite-level FDR control built on top of it is valid.

*Stopping early without flagging* is driven by a boundary crossing or by a
futility bound. Neither can create a false flag, so neither can damage the type-I
guarantee. Futility stopping can cause a miss, and that cost is explicit: it
adds at most ``futility_confidence`` to the per-task miss probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from ..config import StatsConfig
from ..runner.types import RunOutcome, Task
from ..stats import (
    ConfidenceSequence,
    ConfSeqSpec,
    PairedSpec,
    PairedSprt,
    PairOutcome,
    Verdict,
    beta_ppf,
    p_to_e,
)

__all__ = ["StopReason", "TaskResult", "TaskTest"]


class StopReason(StrEnum):
    """Why a task stopped consuming budget."""

    RUNNING = "running"
    BOUNDARY = "boundary"
    FUTILITY = "futility"
    MAX_REPLICATES = "max-replicates"
    BUDGET = "budget"

    @property
    def label(self) -> str:
        return {
            StopReason.RUNNING: "still running",
            StopReason.BOUNDARY: "decided",
            StopReason.FUTILITY: "cannot reach significance in budget",
            StopReason.MAX_REPLICATES: "hit per-task replicate cap",
            StopReason.BUDGET: "suite budget exhausted",
        }[self]


@dataclass
class TaskResult:
    """Everything worth reporting about one task after the gate has run."""

    task_id: str
    verdict: Verdict
    stop_reason: StopReason
    replicates: int
    regressions: int = 0
    improvements: int = 0
    baseline_rate: float = 0.0
    candidate_rate: float = 0.0
    delta: float = 0.0
    e_value: float = 1.0
    anytime_p: float = 1.0
    adjusted_p: float = 1.0
    mcnemar_p: float = 1.0
    flagged: bool = False
    infra_errors: int = 0
    score_mean: float | None = None
    score_interval: tuple[float, float] | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "verdict": self.verdict.value,
            "stop_reason": self.stop_reason.value,
            "replicates": self.replicates,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "baseline_rate": self.baseline_rate,
            "candidate_rate": self.candidate_rate,
            "delta": self.delta,
            "e_value": self.e_value,
            "anytime_p": self.anytime_p,
            "adjusted_p": self.adjusted_p,
            "mcnemar_p": self.mcnemar_p,
            "flagged": self.flagged,
            "infra_errors": self.infra_errors,
            "score_mean": self.score_mean,
            "score_interval": list(self.score_interval) if self.score_interval else None,
            "tags": list(self.tags),
        }


@dataclass
class TaskTest:
    """Sequential evidence for one task, in whichever mode the suite uses."""

    task: Task
    stats: StatsConfig
    alpha_effective: float | None = None
    futility_confidence: float = 0.05
    infra_errors: int = 0
    stop_reason: StopReason = StopReason.RUNNING
    paired: PairedSprt = field(init=False)
    sequence: ConfidenceSequence | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # The stopping boundary is set by what the *suite-level* correction will
        # demand, not by the per-task alpha. Stopping a regressed task the
        # moment it clears its own boundary leaves it holding an e-value of
        # about 18, and a two-hundred-task e-BH correction wants 4000. Stop
        # there and the gate goes permanently blind: every task decides, and
        # none of them survives correction. So the gate passes in a tightened
        # alpha here, and the step-up procedure can only ever flag more than
        # this boundary already justifies.
        alpha = self.alpha_effective if self.alpha_effective is not None else self.stats.alpha
        self.paired = PairedSprt(
            PairedSpec(
                odds_ratio=self.stats.odds_ratio,
                alpha=alpha,
                beta=self.stats.beta,
                max_discordant=self.stats.max_replicates,
                min_discordant=0,
                max_replicates=self.stats.max_replicates,
            )
        )
        if self.stats.mode == "score":
            # Paired differences of scores bounded in [0, range] live in
            # [-range, range], a span of 2*range, so the Hoeffding sub-Gaussian
            # parameter is the range itself.
            self.sequence = ConfidenceSequence(
                ConfSeqSpec(
                    alpha=alpha,
                    sigma=self.stats.score_range,
                    n_opt=max(self.stats.min_replicates * 2, 10),
                )
            )

    @property
    def task_id(self) -> str:
        return self.task.id

    @property
    def replicates(self) -> int:
        return self.paired.replicates

    # -- observation --------------------------------------------------------

    def observe(self, candidate: RunOutcome, baseline: RunOutcome) -> None:
        """Record one paired replicate run under both builds at the same seed."""
        self.paired.observe(
            PairOutcome(candidate_passed=candidate.passed, baseline_passed=baseline.passed)
        )
        has_scores = candidate.score is not None and baseline.score is not None
        if self.sequence is not None and has_scores:
            assert candidate.score is not None and baseline.score is not None
            self.sequence.observe(candidate.score - baseline.score)
        if self.stop_reason is StopReason.RUNNING:
            self._update_stop_reason()

    def record_infra_error(self) -> None:
        """A replicate that could not be run. Counted, never fed to the test."""
        self.infra_errors += 1

    def mark_budget_exhausted(self) -> None:
        if self.stop_reason is StopReason.RUNNING:
            self.stop_reason = StopReason.BUDGET

    def mark_capped(self) -> None:
        """Every replicate slot has been handed out, whether or not it landed.

        Distinct from the check inside :meth:`_update_stop_reason`, which counts
        observed pairs. A task whose replicates all failed to run is capped
        without ever having been observed, and must still stop.
        """
        if self.stop_reason is StopReason.RUNNING:
            self.stop_reason = StopReason.MAX_REPLICATES

    # -- decision -----------------------------------------------------------

    def _update_stop_reason(self) -> None:
        if self._core_verdict() in (Verdict.PASS, Verdict.REGRESSION):
            self.stop_reason = StopReason.BOUNDARY
        elif self.replicates >= self.stats.max_replicates:
            self.stop_reason = StopReason.MAX_REPLICATES
        elif self.replicates >= self.stats.min_replicates and self._is_futile():
            self.stop_reason = StopReason.FUTILITY

    def _core_verdict(self) -> Verdict:
        if self.stats.mode == "score":
            return self._score_verdict()
        return self.paired.verdict

    def _score_verdict(self) -> Verdict:
        assert self.sequence is not None
        threshold = -abs(self.stats.score_tolerance)
        if self.sequence.n < 1:
            return Verdict.CONTINUE
        if self.sequence.is_below(threshold):
            return Verdict.REGRESSION
        if self.sequence.is_above(threshold):
            return Verdict.PASS
        if self.replicates >= self.stats.max_replicates:
            return Verdict.INCONCLUSIVE
        return Verdict.CONTINUE

    def _is_futile(self) -> bool:
        """Can this task still reach the flagging boundary within its cap?

        For the binary mode the bound is honest arithmetic: put a one-sided
        upper confidence bound on how often the two builds disagree, assume
        every future disagreement goes the worst way, and see whether the
        log-likelihood ratio could still clear the boundary. If it cannot, the
        replicates spent here are wasted and belong to another task.
        """
        remaining = self.stats.max_replicates - self.replicates
        if remaining <= 0:
            return True
        if self.stats.mode == "score":
            return self._is_futile_score(remaining)
        inner = self.paired.inner
        d = self.paired.discordant
        r = self.paired.replicates
        # Clopper-Pearson upper bound on the disagreement rate.
        rate_ucb = (
            1.0 if d >= r else beta_ppf(d + 1, r - d, 1.0 - self.futility_confidence)
        )
        best_case_pairs = math.ceil(remaining * rate_ucb)
        best_case_lr = inner.log_lr + best_case_pairs * inner.spec.step_fail
        return best_case_lr < inner.spec.upper

    def _is_futile_score(self, remaining: int) -> bool:
        assert self.sequence is not None
        threshold = -abs(self.stats.score_tolerance)
        n_max = self.sequence.n + remaining
        if n_max < 1:
            return True
        # Best case for detecting a regression: every remaining difference is
        # the most negative one the score range allows.
        floor = -abs(self.stats.score_range)
        best_total = self.sequence.total + remaining * floor
        best_mean = best_total / n_max
        from ..stats.confseq import radius as cs_radius

        best_radius = cs_radius(n_max, self.sequence.spec)
        return (best_mean + best_radius) >= threshold

    @property
    def verdict(self) -> Verdict:
        core = self._core_verdict()
        if core in (Verdict.PASS, Verdict.REGRESSION):
            return core
        if self.stop_reason in (
            StopReason.FUTILITY,
            StopReason.MAX_REPLICATES,
            StopReason.BUDGET,
        ):
            return Verdict.INCONCLUSIVE
        return core

    @property
    def resolved(self) -> bool:
        return self.stop_reason is not StopReason.RUNNING

    @property
    def e_value(self) -> float:
        """Evidence against "this task did not regress", on the e-value scale."""
        if self.stats.mode == "score":
            if self.sequence is None or self.sequence.n < 1:
                return 1.0
            threshold = -abs(self.stats.score_tolerance)
            if self.sequence.mean >= threshold:
                # Wrong direction: no evidence of a regression at all. Clamping
                # to 1 rather than reporting the two-sided evidence keeps the
                # calibration conservative.
                return 1.0
            return p_to_e(self.sequence.anytime_p_value(threshold))
        return self.paired.e_value

    @property
    def anytime_p(self) -> float:
        if self.stats.mode == "score":
            if self.sequence is None or self.sequence.n < 1:
                return 1.0
            threshold = -abs(self.stats.score_tolerance)
            if self.sequence.mean >= threshold:
                return 1.0
            return self.sequence.anytime_p_value(threshold)
        return self.paired.anytime_p_value

    @property
    def priority(self) -> float:
        """How suspicious this task looks, for the allocator's ranking."""
        if self.stats.mode == "score" and self.sequence is not None:
            return -(self.sequence.mean + abs(self.stats.score_tolerance))
        return self.paired.inner.log_lr

    def expected_remaining(self) -> float:
        if self.resolved:
            return 0.0
        if self.stats.mode == "score":
            return float(max(self.stats.max_replicates - self.replicates, 1))
        return self.paired.expected_remaining()

    def result(self) -> TaskResult:
        interval = self.sequence.interval() if self.sequence is not None else None
        return TaskResult(
            task_id=self.task.id,
            verdict=self.verdict,
            stop_reason=self.stop_reason,
            replicates=self.replicates,
            regressions=self.paired.regressions,
            improvements=self.paired.improvements,
            baseline_rate=self.paired.baseline_rate,
            candidate_rate=self.paired.candidate_rate,
            delta=self.paired.delta,
            e_value=self.e_value,
            anytime_p=self.anytime_p,
            mcnemar_p=self.paired.mcnemar_p_value(),
            infra_errors=self.infra_errors,
            score_mean=self.sequence.mean if self.sequence is not None else None,
            score_interval=interval,
            tags=self.task.tags,
        )
