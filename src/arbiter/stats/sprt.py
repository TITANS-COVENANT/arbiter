"""Wald's sequential probability ratio test, specialised to pass/fail evals.

The gate asks one question per task: has the candidate's pass rate dropped by
enough to care? A fixed-sample test answers that by running N replicates and
comparing proportions. The SPRT answers it by watching the log-likelihood ratio
after every replicate and stopping the moment it crosses a boundary, which for
the effect sizes people actually care about costs roughly half the samples.

Two guarantees come out of the construction, and both are checked by Monte Carlo
in ``tests/test_stats_core.py``:

* the probability of declaring a regression when the candidate is fine is at
  most ``alpha``;
* the probability of missing a regression of size ``p0 - p1`` is at most
  ``beta``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = ["Sprt", "SprtSpec", "Verdict"]


class Verdict(StrEnum):
    """Outcome of a sequential test at the current sample count."""

    CONTINUE = "continue"
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"

    @property
    def is_terminal(self) -> bool:
        return self is not Verdict.CONTINUE


@dataclass(frozen=True)
class SprtSpec:
    """Hypotheses and error budget for one task.

    ``p0`` is the pass rate under "nothing changed" and ``p1`` the pass rate the
    test is powered to catch. Everything between the two is a grey zone the test
    is explicitly not asked to resolve, which is what keeps the sample count
    finite. You cannot detect an arbitrarily small regression cheaply, so you
    have to say out loud how small is small enough to ignore.
    """

    p0: float
    p1: float
    alpha: float = 0.05
    beta: float = 0.10
    max_samples: int = 200
    min_samples: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.p1 < self.p0 < 1.0:
            raise ValueError(
                f"need 0 < p1 < p0 < 1 (p1 is the regressed rate), "
                f"got p0={self.p0}, p1={self.p1}"
            )
        if not 0.0 < self.alpha < 0.5 or not 0.0 < self.beta < 0.5:
            raise ValueError(f"alpha and beta must lie in (0, 0.5), got {self.alpha}, {self.beta}")
        if self.max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        if self.min_samples < 0 or self.min_samples > self.max_samples:
            raise ValueError("min_samples must lie in [0, max_samples]")

    @classmethod
    def from_mde(
        cls,
        baseline_rate: float,
        mde: float,
        *,
        alpha: float = 0.05,
        beta: float = 0.10,
        max_samples: int = 200,
        min_samples: int = 0,
    ) -> SprtSpec:
        """Build a spec from a baseline rate and a minimum detectable effect.

        ``mde=0.15`` on a baseline of 0.90 means "catch it if the pass rate falls
        to 0.75 or below". Rates are clamped away from 0 and 1 because the
        log-likelihood ratio is undefined at the boundary.
        """
        if not 0.0 < mde < 1.0:
            raise ValueError(f"mde must lie in (0, 1), got {mde}")
        p0 = min(max(baseline_rate, 2e-6), 1.0 - 1e-6)
        p1 = min(max(p0 - mde, 1e-6), p0 - 1e-6)
        return cls(
            p0=p0,
            p1=p1,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples,
            min_samples=min_samples,
        )

    @property
    def upper(self) -> float:
        """Log-likelihood-ratio boundary above which H1 (regression) is accepted."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        """Log-likelihood-ratio boundary below which H0 (no regression) is accepted."""
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def step_pass(self) -> float:
        """Log-LR increment contributed by one passing replicate (negative)."""
        return math.log(self.p1 / self.p0)

    @property
    def step_fail(self) -> float:
        """Log-LR increment contributed by one failing replicate (positive)."""
        return math.log((1.0 - self.p1) / (1.0 - self.p0))

    def drift(self, p: float) -> float:
        """Expected log-LR increment per replicate when the true rate is ``p``."""
        return p * self.step_pass + (1.0 - p) * self.step_fail

    def variance(self, p: float) -> float:
        """Per-replicate variance of the log-LR increment at true rate ``p``."""
        mu = self.drift(p)
        second = p * self.step_pass**2 + (1.0 - p) * self.step_fail**2
        return max(second - mu**2, 1e-12)

    def operating_characteristic(self, p: float) -> tuple[float, float]:
        """Return ``(P(accept H0), E[samples])`` when the true pass rate is ``p``.

        Uses Wald's approximation, which ignores boundary overshoot and so runs a
        few percent optimistic on the sample count. It is used for planning and
        for the scheduler's priority estimate, never for the decision itself.
        """
        mu = self.drift(p)
        h = self._solve_h(p)
        if h is None:
            # Zero drift: the walk is a martingale, so the acceptance
            # probability is the ratio of the distances to each boundary and the
            # expected duration is the product of them over the step variance.
            accept_h0 = self.upper / (self.upper - self.lower)
            expected_n = -self.upper * self.lower / self.variance(p)
        else:
            a_h = math.exp(h * self.upper)
            b_h = math.exp(h * self.lower)
            accept_h0 = (a_h - 1.0) / (a_h - b_h)
            accept_h0 = min(max(accept_h0, 0.0), 1.0)
            expected_n = ((1.0 - accept_h0) * self.upper + accept_h0 * self.lower) / mu
        accept_h0 = min(max(accept_h0, 0.0), 1.0)
        expected_n = min(max(expected_n, 1.0), float(self.max_samples))
        return accept_h0, expected_n

    def _solve_h(self, p: float) -> float | None:
        """Solve ``E_p[exp(h * z)] = 1`` for the non-zero root, by bisection.

        Returns ``None`` when the drift is negligible, in which case the root
        collapses onto zero and the caller should use the martingale formulas.
        """
        mu = self.drift(p)
        if abs(mu) < 1e-9:
            return None

        def f(h: float) -> float:
            return (
                p * math.exp(h * self.step_pass)
                + (1.0 - p) * math.exp(h * self.step_fail)
                - 1.0
            )

        # f(0) = 0 and f is convex, so the second root sits on the side opposite
        # the drift. Walk outward until the sign flips, then bisect.
        direction = 1.0 if mu < 0 else -1.0
        lo = 0.0
        hi = direction * 1e-3
        for _ in range(200):
            if f(hi) > 0.0:
                break
            lo = hi
            hi *= 2.0
            if abs(hi) > 1e4:
                return None
        else:
            return None
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) < 1e-14:
                break
        root = 0.5 * (lo + hi)
        return None if abs(root) < 1e-9 else root


@dataclass
class Sprt:
    """Running state of one task's sequential test."""

    spec: SprtSpec
    n: int = 0
    passes: int = 0
    log_lr: float = 0.0
    max_log_lr: float = 0.0
    history: list[float] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return self.n - self.passes

    @property
    def rate(self) -> float:
        return self.passes / self.n if self.n else 0.0

    @property
    def smoothed_rate(self) -> float:
        """Laplace-smoothed pass rate, safe to use as a plug-in estimate at n=0."""
        return (self.passes + 0.5) / (self.n + 1.0)

    def observe(self, passed: bool) -> Verdict:
        """Record one replicate and return the verdict as of this sample."""
        self.n += 1
        if passed:
            self.passes += 1
            self.log_lr += self.spec.step_pass
        else:
            self.log_lr += self.spec.step_fail
        self.max_log_lr = max(self.max_log_lr, self.log_lr)
        self.history.append(self.log_lr)
        return self.verdict

    def observe_many(self, outcomes: list[bool]) -> Verdict:
        """Feed a batch of replicates, stopping at the first terminal verdict."""
        verdict = self.verdict
        for outcome in outcomes:
            if verdict.is_terminal:
                break
            verdict = self.observe(outcome)
        return verdict

    @property
    def verdict(self) -> Verdict:
        if self.n < self.spec.min_samples:
            return Verdict.CONTINUE
        if self.log_lr >= self.spec.upper:
            return Verdict.REGRESSION
        if self.log_lr <= self.spec.lower:
            return Verdict.PASS
        if self.n >= self.spec.max_samples:
            return Verdict.INCONCLUSIVE
        return Verdict.CONTINUE

    @property
    def e_value(self) -> float:
        """The likelihood ratio, a valid e-value for H0 at any stopping time.

        Under the composite null "the pass rate is at least p0" the per-sample
        expectation of the likelihood ratio is at most 1, so the running product
        is a non-negative supermartingale starting at 1. Optional stopping then
        gives ``E[e_value] <= 1``, which is exactly the property e-BH needs in
        :mod:`arbiter.stats.multiple`.
        """
        return math.exp(min(self.log_lr, 700.0))

    @property
    def anytime_p_value(self) -> float:
        """Ville's inequality applied to the running maximum likelihood ratio.

        Safe to inspect after every replicate, unlike a fixed-sample p-value.
        That is the whole reason peeking at eval results is normally a
        statistical crime and is not one here.
        """
        return min(1.0, math.exp(-min(self.max_log_lr, 700.0)))

    def expected_remaining(self, p_hat: float | None = None) -> float:
        """Cheap estimate of how many more replicates until this task decides.

        The scheduler uses this to spend its budget where it closes decisions
        fastest, so it needs to be cheap rather than exact.
        """
        if self.verdict.is_terminal:
            return 0.0
        p = self.smoothed_rate if p_hat is None else p_hat
        mu = self.spec.drift(p)
        budget_left = float(self.spec.max_samples - self.n)
        if abs(mu) < 1e-9:
            return budget_left
        target = self.spec.upper if mu > 0 else self.spec.lower
        remaining = (target - self.log_lr) / mu
        return min(max(remaining, 1.0), budget_left)

    def snapshot(self) -> dict[str, object]:
        return {
            "n": self.n,
            "passes": self.passes,
            "rate": self.rate,
            "log_lr": self.log_lr,
            "e_value": self.e_value,
            "anytime_p": self.anytime_p_value,
            "verdict": self.verdict.value,
        }
