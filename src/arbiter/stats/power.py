"""Planning: how many runs, how many dollars, how many minutes.

``arbiter plan`` exists so that the sample budget is a decision someone makes on
purpose rather than a number that got copied from another repo. Given a baseline
pass rate, the smallest regression worth catching, and what a replicate costs,
it reports what the fixed-sample approach would spend and what the sequential
one is expected to spend.

Everything here is an estimate under Wald's approximations, and the numbers come
out a few percent optimistic on sample count because boundary overshoot is
ignored. The simulator in :mod:`arbiter.sim` measures the real thing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .special import norm_ppf
from .sprt import SprtSpec

__all__ = [
    "PairedPlan",
    "SuitePlan",
    "TaskPlan",
    "fixed_sample_size",
    "paired_plan",
    "plan_suite",
]


def fixed_sample_size(p0: float, p1: float, alpha: float = 0.05, beta: float = 0.10) -> int:
    """Replicates needed by a one-sided fixed-sample test of p0 against p1.

    The normal approximation to the binomial, which is what every sample-size
    calculator uses and is accurate enough for planning at these rates.
    """
    if not 0.0 < p1 < p0 < 1.0:
        raise ValueError(f"need 0 < p1 < p0 < 1, got p0={p0}, p1={p1}")
    z_alpha = norm_ppf(1.0 - alpha)
    z_beta = norm_ppf(1.0 - beta)
    numerator = z_alpha * (p0 * (1.0 - p0)) ** 0.5 + z_beta * (p1 * (1.0 - p1)) ** 0.5
    return max(1, int((numerator / (p0 - p1)) ** 2 + 0.999))


@dataclass(frozen=True)
class PairedPlan:
    """What the paired sequential test needs, which is what arbiter actually runs.

    This is the number to size ``max_replicates`` from. It is usually larger
    than the fixed-sample figure people expect, for two reasons that are both
    real rather than artefacts of the method.

    First, only replicates where the two builds disagree carry information about
    the change, and if they agree 90% of the time then 90% of the runs are
    telling you about the task rather than about the diff.

    Second, suite-level false discovery control raises the bar every task has to
    clear. A single regression hiding in two hundred tasks has to be much more
    obvious than one hiding in five, and no amount of cleverness removes that.
    """

    n_tasks: int
    evidence_threshold: float
    discordant_pairs_needed: float
    discordance_rate: float
    implied_odds_ratio: float = 0.0

    @property
    def replicates_needed(self) -> float:
        """Replicates per task, both builds counted once each."""
        return self.discordant_pairs_needed / max(self.discordance_rate, 1e-9)

    @property
    def total_replicates(self) -> float:
        return self.replicates_needed * self.n_tasks * 2


def paired_plan(
    *,
    n_tasks: int,
    baseline_rate: float,
    mde: float,
    coupling: float = 0.7,
    alpha: float = 0.05,
    beta: float = 0.10,
    odds_ratio: float = 3.0,
    correction: str = "e-bh",
) -> PairedPlan:
    """Size a suite for the paired sequential test.

    ``coupling`` is the share of replicates on which fixing the seed makes the
    two builds behave identically. It is the single biggest lever on cost and
    almost nobody measures it, so the default of 0.7 is a guess: run
    ``arbiter gate`` once and the report tells you the real disagreement rate.
    """
    from .sprt import SprtSpec

    threshold = (1.0 / alpha) if correction == "none" else (n_tasks / alpha)
    alpha_effective = min(max((1.0 - beta) / threshold, 1e-12), alpha)
    spec = SprtSpec(
        p0=0.5,
        p1=1.0 / (1.0 + odds_ratio),
        alpha=alpha_effective,
        beta=beta,
        max_samples=10**6,
    )

    # Expected disagreement rate for a task that really did drop by `mde`: the
    # coupled replicates disagree exactly where the two rates straddle the
    # shared latent draw, and the uncoupled ones disagree independently.
    candidate_rate = max(baseline_rate - mde, 0.0)
    regressions = coupling * mde + (1.0 - coupling) * baseline_rate * (1.0 - candidate_rate)
    improvements = (1.0 - coupling) * candidate_rate * (1.0 - baseline_rate)
    discordance = regressions + improvements
    if discordance <= 0:
        return PairedPlan(n_tasks, threshold, math.inf, 0.0, 0.0)

    # The declared odds ratio sets where the boundary sits; the *true* split
    # decides how fast the walk gets there. Using the declared one for both
    # would say that a target which honours its seed needs more replicates,
    # which is backwards: tight coupling means almost every disagreement is a
    # genuine regression, and those are the informative ones.
    improvement_share = improvements / discordance
    drift = improvement_share * spec.step_pass + (1.0 - improvement_share) * spec.step_fail
    pairs = spec.upper / drift if drift > 0 else math.inf
    implied_odds = regressions / improvements if improvements > 0 else math.inf
    return PairedPlan(
        n_tasks=n_tasks,
        evidence_threshold=threshold,
        discordant_pairs_needed=pairs,
        discordance_rate=discordance,
        implied_odds_ratio=implied_odds,
    )


@dataclass(frozen=True)
class TaskPlan:
    """Expected cost of resolving a single task."""

    fixed_samples: int
    expected_samples_under_null: float
    expected_samples_under_alt: float

    @property
    def expected_samples(self) -> float:
        """Cost on a typical build, where almost every task is unchanged.

        Weighted heavily toward the null because that is the honest expectation:
        most CI runs do not contain a regression, and a planner that quotes the
        alternative-hypothesis cost is quoting the rare case.
        """
        return 0.9 * self.expected_samples_under_null + 0.1 * self.expected_samples_under_alt

    @property
    def savings(self) -> float:
        if not self.fixed_samples:
            return 0.0
        return 1.0 - self.expected_samples / self.fixed_samples


@dataclass(frozen=True)
class SuitePlan:
    """Expected cost of gating a whole suite."""

    n_tasks: int
    per_task: TaskPlan
    cost_per_replicate: float
    seconds_per_replicate: float
    concurrency: int

    @property
    def fixed_replicates(self) -> int:
        return self.n_tasks * self.per_task.fixed_samples

    @property
    def expected_replicates(self) -> float:
        return self.n_tasks * self.per_task.expected_samples

    @property
    def fixed_cost(self) -> float:
        return self.fixed_replicates * self.cost_per_replicate

    @property
    def expected_cost(self) -> float:
        return self.expected_replicates * self.cost_per_replicate

    @property
    def fixed_wall_clock_seconds(self) -> float:
        return self.fixed_replicates * self.seconds_per_replicate / max(self.concurrency, 1)

    @property
    def expected_wall_clock_seconds(self) -> float:
        return self.expected_replicates * self.seconds_per_replicate / max(self.concurrency, 1)

    @property
    def savings(self) -> float:
        return self.per_task.savings


def plan_suite(
    *,
    n_tasks: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    max_samples: int = 200,
    cost_per_replicate: float = 0.0,
    seconds_per_replicate: float = 0.0,
    concurrency: int = 1,
) -> SuitePlan:
    """Estimate what gating a suite of ``n_tasks`` will cost.

    Note that the per-task error budget is *not* divided by the number of tasks
    here. Suite-level error control is handled by the FDR step in
    :mod:`arbiter.stats.multiple`, which is far less punishing on sample counts
    than a Bonferroni split would be.
    """
    if n_tasks < 1:
        raise ValueError("n_tasks must be at least 1")
    spec = SprtSpec.from_mde(
        baseline_rate, mde, alpha=alpha, beta=beta, max_samples=max_samples
    )
    _, expected_null = spec.operating_characteristic(spec.p0)
    _, expected_alt = spec.operating_characteristic(spec.p1)
    per_task = TaskPlan(
        fixed_samples=fixed_sample_size(spec.p0, spec.p1, alpha, beta),
        expected_samples_under_null=expected_null,
        expected_samples_under_alt=expected_alt,
    )
    return SuitePlan(
        n_tasks=n_tasks,
        per_task=per_task,
        cost_per_replicate=cost_per_replicate,
        seconds_per_replicate=seconds_per_replicate,
        concurrency=concurrency,
    )
