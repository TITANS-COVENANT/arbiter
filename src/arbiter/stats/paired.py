"""Sequential paired comparison of a candidate against a baseline.

Most of the variance in an agent eval is the task, not the change under test.
Task 41 is hard and fails half the time for both builds; task 12 is easy and
passes for both. Comparing two independent pass-rate estimates makes you pay for
that task-level variance twice, and it swamps the effect you are looking for.

So arbiter runs both builds on the *same* task with the *same* seed and looks
only at the pairs where they disagree. Under the null that the change did
nothing, a disagreement is equally likely to fall either way, so the discordant
pairs are fair coin flips. That is McNemar's test, and running it sequentially
turns it into an SPRT on a Bernoulli(0.5) null.

Concordant pairs carry no information about the *difference* and are dropped,
which is the mathematical version of the intuition that a task both builds fail
tells you nothing about which build is better.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .sprt import Sprt, SprtSpec, Verdict

__all__ = ["PairOutcome", "PairedSpec", "PairedSprt"]


@dataclass(frozen=True)
class PairOutcome:
    """One replicate run under both builds at the same seed."""

    candidate_passed: bool
    baseline_passed: bool

    @property
    def is_concordant(self) -> bool:
        return self.candidate_passed == self.baseline_passed

    @property
    def is_regression(self) -> bool:
        """Baseline passed where the candidate failed."""
        return self.baseline_passed and not self.candidate_passed

    @property
    def is_improvement(self) -> bool:
        """Candidate passed where the baseline failed."""
        return self.candidate_passed and not self.baseline_passed


@dataclass(frozen=True)
class PairedSpec:
    """Error budget and effect size for a paired sequential comparison.

    The effect size is stated as an odds ratio on discordant pairs.
    ``odds_ratio=3.0`` means "call it a regression if failures introduced by the
    candidate outnumber failures it fixed by three to one". That is a more
    natural thing to specify than an absolute rate drop, because it is exactly
    what a reviewer looking at the diff would count.
    """

    odds_ratio: float = 3.0
    alpha: float = 0.05
    beta: float = 0.10
    max_discordant: int = 60
    min_discordant: int = 0
    max_replicates: int = 200

    def __post_init__(self) -> None:
        if self.odds_ratio <= 1.0:
            raise ValueError(f"odds_ratio must exceed 1, got {self.odds_ratio}")
        if self.max_replicates < 1:
            raise ValueError("max_replicates must be at least 1")

    def to_sprt_spec(self) -> SprtSpec:
        """Recast as an SPRT on the fraction of discordant pairs that are improvements.

        Under the null that fraction is 1/2. Under the alternative it falls to
        ``1 / (1 + odds_ratio)``. Framing it as the *improvement* fraction rather
        than the regression fraction keeps the "pass rate goes down is bad"
        orientation that :class:`~arbiter.stats.sprt.SprtSpec` expects.
        """
        return SprtSpec(
            p0=0.5,
            p1=1.0 / (1.0 + self.odds_ratio),
            alpha=self.alpha,
            beta=self.beta,
            max_samples=self.max_discordant,
            min_samples=self.min_discordant,
        )


@dataclass
class PairedSprt:
    """Running state of one task's paired sequential comparison."""

    spec: PairedSpec
    inner: Sprt = field(init=False)
    replicates: int = 0
    regressions: int = 0
    improvements: int = 0
    both_passed: int = 0
    both_failed: int = 0

    def __post_init__(self) -> None:
        self.inner = Sprt(self.spec.to_sprt_spec())

    @property
    def discordant(self) -> int:
        return self.regressions + self.improvements

    @property
    def discordance_rate(self) -> float:
        """Smoothed fraction of replicates that disagree between the builds.

        Used to translate "how many more discordant pairs do I need" into "how
        many more replicates must I actually run", which is what the scheduler
        is spending.
        """
        return (self.discordant + 0.5) / (self.replicates + 1.0)

    @property
    def candidate_rate(self) -> float:
        passed = self.both_passed + self.improvements
        return passed / self.replicates if self.replicates else 0.0

    @property
    def baseline_rate(self) -> float:
        passed = self.both_passed + self.regressions
        return passed / self.replicates if self.replicates else 0.0

    @property
    def delta(self) -> float:
        """Estimated change in pass rate, candidate minus baseline."""
        if not self.replicates:
            return 0.0
        return (self.improvements - self.regressions) / self.replicates

    def observe(self, outcome: PairOutcome) -> Verdict:
        """Record one paired replicate and return the verdict as of this pair."""
        self.replicates += 1
        if outcome.is_regression:
            self.regressions += 1
            self.inner.observe(False)
        elif outcome.is_improvement:
            self.improvements += 1
            self.inner.observe(True)
        elif outcome.candidate_passed:
            self.both_passed += 1
        else:
            self.both_failed += 1
        return self.verdict

    @property
    def verdict(self) -> Verdict:
        inner_verdict = self.inner.verdict
        if inner_verdict.is_terminal:
            return inner_verdict
        if self.replicates >= self.spec.max_replicates:
            return Verdict.INCONCLUSIVE
        return Verdict.CONTINUE

    @property
    def e_value(self) -> float:
        return self.inner.e_value

    @property
    def anytime_p_value(self) -> float:
        return self.inner.anytime_p_value

    def expected_remaining(self) -> float:
        """Estimated additional *replicates* needed, not discordant pairs.

        A task where both builds behave identically produces almost no
        discordant pairs, so it can be arbitrarily expensive to resolve. The
        scheduler needs to see that cost, which is why this divides through by
        the observed discordance rate rather than reporting the inner test's
        remaining pairs directly.
        """
        if self.verdict.is_terminal:
            return 0.0
        pairs_needed = self.inner.expected_remaining()
        replicates_needed = pairs_needed / max(self.discordance_rate, 1e-6)
        budget_left = float(self.spec.max_replicates - self.replicates)
        return min(max(replicates_needed, 1.0), budget_left)

    def mcnemar_p_value(self) -> float:
        """Two-sided exact McNemar p-value, for reporting alongside the verdict.

        This is a fixed-sample quantity and is *not* what the gate decides on.
        It is here because reviewers ask for it and because it is a useful
        sanity check against the sequential result at the stopping time.
        """
        b, c = self.regressions, self.improvements
        m = b + c
        if m == 0:
            return 1.0
        k = min(b, c)
        tail = sum(math.comb(m, i) for i in range(k + 1)) / (2.0**m)
        return min(1.0, 2.0 * tail)

    def snapshot(self) -> dict[str, object]:
        return {
            "replicates": self.replicates,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "both_passed": self.both_passed,
            "both_failed": self.both_failed,
            "delta": self.delta,
            "e_value": self.e_value,
            "anytime_p": self.anytime_p_value,
            "mcnemar_p": self.mcnemar_p_value(),
            "verdict": self.verdict.value,
        }
