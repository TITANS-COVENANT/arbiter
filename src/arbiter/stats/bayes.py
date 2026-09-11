"""Beta-binomial posteriors and a Bayesian stopping rule.

The frequentist tests answer "can I reject the null at my error budget". Some
teams would rather answer "what is the probability this build is worse, and by
how much", which is a different question and often the one the person deciding
whether to merge actually has.

Both are supported; the gate picks between them by policy. The Bayesian rule
stops when the posterior probability of a regression exceeds a threshold, which
has no frequentist error guarantee but is easy to explain and behaves well when
the prior genuinely encodes what you know from previous runs of the same suite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .special import beta_ppf, betainc, log_beta

__all__ = ["BetaPosterior", "prob_worse", "prob_worse_by"]


@dataclass(frozen=True)
class BetaPosterior:
    """Posterior over a pass rate after observing binomial data."""

    alpha: float
    beta: float

    @classmethod
    def from_data(
        cls,
        passes: int,
        n: int,
        *,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> BetaPosterior:
        if passes < 0 or n < passes:
            raise ValueError(f"need 0 <= passes <= n, got passes={passes}, n={n}")
        return cls(prior_alpha + passes, prior_beta + (n - passes))

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return self.alpha * self.beta / (s * s * (s + 1.0))

    def cdf(self, x: float) -> float:
        return betainc(self.alpha, self.beta, x)

    def quantile(self, q: float) -> float:
        return beta_ppf(self.alpha, self.beta, q)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval."""
        tail = (1.0 - level) / 2.0
        return (self.quantile(tail), self.quantile(1.0 - tail))

    def log_pdf(self, x: float) -> float:
        if not 0.0 < x < 1.0:
            return -math.inf
        return (
            (self.alpha - 1.0) * math.log(x)
            + (self.beta - 1.0) * math.log1p(-x)
            - log_beta(self.alpha, self.beta)
        )


def prob_worse_by(
    baseline: BetaPosterior,
    candidate: BetaPosterior,
    delta: float = 0.0,
    *,
    grid: int = 2000,
) -> float:
    """P(candidate rate < baseline rate - delta) under independent posteriors.

    Evaluated by Simpson quadrature over the baseline posterior:

        integral over x of  f_baseline(x) * F_candidate(x - delta) dx

    A closed form exists for delta=0 as a finite sum, but it needs integer
    parameters and blows up combinatorially for large n, whereas quadrature is
    uniform in cost and handles the non-zero tolerance that anyone gating on a
    real suite will want.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must lie in [0, 1), got {delta}")
    if grid % 2:
        grid += 1

    lo = delta + 1e-9
    hi = 1.0 - 1e-9
    if lo >= hi:
        return 0.0
    step = (hi - lo) / grid

    def integrand(x: float) -> float:
        return math.exp(baseline.log_pdf(x)) * candidate.cdf(x - delta)

    total = integrand(lo) + integrand(hi)
    for i in range(1, grid):
        weight = 4.0 if i % 2 else 2.0
        total += weight * integrand(lo + i * step)
    return min(max(total * step / 3.0, 0.0), 1.0)


def prob_worse(
    baseline_passes: int,
    baseline_n: int,
    candidate_passes: int,
    candidate_n: int,
    delta: float = 0.0,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    """Convenience wrapper over :func:`prob_worse_by` taking raw counts."""
    baseline = BetaPosterior.from_data(
        baseline_passes, baseline_n, prior_alpha=prior_alpha, prior_beta=prior_beta
    )
    candidate = BetaPosterior.from_data(
        candidate_passes, candidate_n, prior_alpha=prior_alpha, prior_beta=prior_beta
    )
    return prob_worse_by(baseline, candidate, delta)
