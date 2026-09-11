"""Monte Carlo harness: measure what the gate actually does, not what it claims.

Everything in the README that is a number came out of here. The synthetic agent
has a known truth, so a few hundred trials give an honest estimate of the false
discovery rate, the power, and what it all cost, which is the only way to tell
whether the theory survived contact with the implementation.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..config import (
    BudgetConfig,
    GateConfig,
    StatsConfig,
    SuiteConfig,
    TargetConfig,
    TaskConfig,
)
from ..gate.decide import run_gate
from ..stats import fixed_sample_size

__all__ = ["ExperimentResult", "ScenarioSpec", "TrialOutcome", "make_suite", "run_experiment"]


@dataclass(frozen=True)
class ScenarioSpec:
    """One experimental condition."""

    n_tasks: int = 40
    n_regressed: int = 4
    baseline_rate: float = 0.90
    regressed_rate: float = 0.70
    coupling: float = 0.7
    alpha: float = 0.05
    beta: float = 0.10
    mde: float = 0.15
    odds_ratio: float = 3.0
    min_replicates: int = 4
    max_replicates: int = 80
    correction: str = "e-bh"
    allocator: str = "cheapest-to-close"
    batch_size: int = 32
    max_replicates_budget: int | None = None

    @property
    def regressed_ids(self) -> set[str]:
        return {f"task_{i:03d}" for i in range(self.n_regressed)}


def make_suite(spec: ScenarioSpec, salt: str) -> SuiteConfig:
    """Build a suite whose ground truth is known.

    The first ``n_regressed`` tasks really did get worse; the rest really did
    not. ``salt`` changes every seed in the suite, which is how independent
    trials are drawn without touching the simulator's internals.
    """
    tasks = []
    for i in range(spec.n_tasks):
        regressed = i < spec.n_regressed
        tasks.append(
            TaskConfig(
                id=f"task_{i:03d}",
                tags=["regressed"] if regressed else ["clean"],
                input={
                    "baseline_rate": spec.baseline_rate,
                    "candidate_rate": spec.regressed_rate if regressed else spec.baseline_rate,
                    "coupling": spec.coupling,
                    "cost_usd": 0.02,
                    "failure_mode": i % 5,
                    "candidate_failure_mode": 2 if regressed else None,
                },
            )
        )
    return SuiteConfig(
        name="simulated",
        seed_salt=salt,
        tasks=tasks,
        baseline=TargetConfig(kind="python", ref="arbiter.sim.agent:baseline", concurrency=64),
        candidate=TargetConfig(kind="python", ref="arbiter.sim.agent:candidate", concurrency=64),
        stats=StatsConfig(
            alpha=spec.alpha,
            beta=spec.beta,
            mde=spec.mde,
            odds_ratio=spec.odds_ratio,
            min_replicates=spec.min_replicates,
            max_replicates=spec.max_replicates,
            correction=spec.correction,  # type: ignore[arg-type]
        ),
        budget=BudgetConfig(
            batch_size=spec.batch_size, max_replicates=spec.max_replicates_budget
        ),
        gate=GateConfig(allocator=spec.allocator),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class TrialOutcome:
    """What one simulated gate run produced."""

    true_positives: int
    false_positives: int
    n_flagged: int
    replicates: int
    cost_usd: float
    verdict: str
    caught_any: bool

    @property
    def false_discovery_proportion(self) -> float:
        return self.false_positives / self.n_flagged if self.n_flagged else 0.0


@dataclass
class ExperimentResult:
    """Aggregate over trials, with the numbers worth quoting."""

    spec: ScenarioSpec
    trials: list[TrialOutcome] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def fdr(self) -> float:
        """Realised false discovery rate: the mean false-discovery proportion.

        This is the quantity e-BH bounds by alpha, and the mean is taken over
        every trial including those that flagged nothing (which contribute
        zero), because that is how the FDR is defined.
        """
        if not self.trials:
            return 0.0
        return statistics.fmean(t.false_discovery_proportion for t in self.trials)

    @property
    def per_task_power(self) -> float:
        """Share of genuinely regressed tasks that got flagged."""
        if not self.trials or not self.spec.n_regressed:
            return 0.0
        total = self.spec.n_regressed * self.n_trials
        return sum(t.true_positives for t in self.trials) / total

    @property
    def detection_rate(self) -> float:
        """Share of trials where the gate caught at least one real regression.

        The number a team actually cares about: did the bad build get blocked.
        """
        if not self.trials:
            return 0.0
        return statistics.fmean(float(t.caught_any) for t in self.trials)

    @property
    def any_false_flag_rate(self) -> float:
        """Share of trials with at least one false flag: how often it cries wolf."""
        if not self.trials:
            return 0.0
        return statistics.fmean(float(t.false_positives > 0) for t in self.trials)

    @property
    def mean_replicates(self) -> float:
        return statistics.fmean(t.replicates for t in self.trials) if self.trials else 0.0

    @property
    def mean_cost(self) -> float:
        return statistics.fmean(t.cost_usd for t in self.trials) if self.trials else 0.0

    @property
    def fixed_replicates_uncorrected(self) -> int:
        """What a fixed-sample design costs at a per-task alpha, with no correction.

        This is what most eval suites actually do: pick N, run everything N
        times, compare each task at 0.05. It is the cheaper number, and it does
        not buy the suite-level guarantee, so quoting arbiter's saving against
        it would be comparing two different products.
        """
        per_task = fixed_sample_size(
            self.spec.baseline_rate,
            max(self.spec.baseline_rate - self.spec.mde, 1e-6),
            self.spec.alpha,
            self.spec.beta,
        )
        return per_task * self.spec.n_tasks * 2

    @property
    def fixed_replicates(self) -> int:
        """What a fixed-sample design costs at the *same* suite-level guarantee.

        To bound false discoveries across the suite without sequential
        machinery you split alpha across the tasks, which raises the per-task
        sample size. Doubled because arbiter runs both builds and so would any
        honest fixed-sample comparison against a fresh baseline.
        """
        if self.spec.correction == "none":
            return self.fixed_replicates_uncorrected
        alpha = self.spec.alpha / self.spec.n_tasks
        per_task = fixed_sample_size(
            self.spec.baseline_rate,
            max(self.spec.baseline_rate - self.spec.mde, 1e-6),
            alpha,
            self.spec.beta,
        )
        return per_task * self.spec.n_tasks * 2

    @property
    def savings_vs_uncorrected(self) -> float:
        if not self.fixed_replicates_uncorrected:
            return 0.0
        return 1.0 - self.mean_replicates / self.fixed_replicates_uncorrected

    @property
    def savings(self) -> float:
        if not self.fixed_replicates:
            return 0.0
        return 1.0 - self.mean_replicates / self.fixed_replicates

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "n_tasks": self.spec.n_tasks,
            "n_regressed": self.spec.n_regressed,
            "baseline_rate": self.spec.baseline_rate,
            "regressed_rate": self.spec.regressed_rate,
            "coupling": self.spec.coupling,
            "correction": self.spec.correction,
            "allocator": self.spec.allocator,
            "alpha": self.spec.alpha,
            "fdr": self.fdr,
            "per_task_power": self.per_task_power,
            "detection_rate": self.detection_rate,
            "any_false_flag_rate": self.any_false_flag_rate,
            "mean_replicates": self.mean_replicates,
            "fixed_replicates": self.fixed_replicates,
            "fixed_replicates_uncorrected": self.fixed_replicates_uncorrected,
            "savings": self.savings,
            "savings_vs_uncorrected": self.savings_vs_uncorrected,
            "mean_cost_usd": self.mean_cost,
        }


async def run_experiment(
    spec: ScenarioSpec,
    n_trials: int = 100,
    *,
    salt_prefix: str = "trial",
    concurrency: int = 4,
) -> ExperimentResult:
    """Run ``n_trials`` independent gate runs against the synthetic agent."""
    result = ExperimentResult(spec=spec)
    regressed = spec.regressed_ids
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> TrialOutcome:
        async with semaphore:
            suite = make_suite(spec, f"{salt_prefix}-{index}")
            gate = await run_gate(suite)
        flagged = {t.task_id for t in gate.tasks if t.flagged}
        true_positives = len(flagged & regressed)
        return TrialOutcome(
            true_positives=true_positives,
            false_positives=len(flagged - regressed),
            n_flagged=len(flagged),
            replicates=gate.replicates_run,
            cost_usd=gate.cost_usd,
            verdict=gate.verdict,
            caught_any=true_positives > 0,
        )

    result.trials = list(await asyncio.gather(*(one(i) for i in range(n_trials))))
    return result
