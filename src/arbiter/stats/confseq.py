"""Always-valid confidence sequences for scalar eval scores.

Not every eval is pass/fail. Rubric scores, BLEU-alikes, latency budgets and
cost-per-task are all continuous, and for those the question is not "did the
rate drop" but "where is the mean, and can I stop looking yet".

A fixed-sample confidence interval answers that only if you commit to N in
advance. Look at it after every batch and stop when it excludes zero and you
have re-invented p-hacking. A confidence sequence is an interval that holds
*simultaneously at every sample size*, so you can watch it and stop whenever you
like without inflating the error rate. The price is width: roughly 1.9x a
fixed-sample interval at the sample size it is tuned for.

The construction here is the Gaussian mixture boundary of Robbins, in the form
popularised for always-valid A/B testing. ``tests/test_stats_core.py`` checks
coverage empirically by running streams and counting how often the true mean
ever escapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ConfSeqSpec", "ConfidenceSequence", "radius"]


@dataclass(frozen=True)
class ConfSeqSpec:
    """Tuning for a confidence sequence.

    ``sigma`` is the sub-Gaussian parameter of a single observation. For scores
    bounded in [0, 1] the Hoeffding bound gives 1/2, which is conservative but
    always valid. For paired differences bounded in [-1, 1] use 1.0.

    ``n_opt`` is where the boundary is tightest. The sequence is valid at every
    n regardless, but it is worth setting this near the sample size you expect
    to stop at, because a mixture tuned for n=500 is needlessly wide at n=20.
    """

    alpha: float = 0.05
    sigma: float = 0.5
    n_opt: int = 50

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_opt < 1:
            raise ValueError(f"n_opt must be at least 1, got {self.n_opt}")


def radius(n: int, spec: ConfSeqSpec) -> float:
    """Half-width of the confidence sequence after ``n`` observations.

    Substituting a mixture variance of ``sigma^2 / n_opt`` into Robbins' normal
    mixture boundary collapses it to

        sigma * sqrt( 2 (n + n_opt) / n^2 * log( sqrt((n + n_opt) / n_opt) / alpha ) )

    which decays like sqrt(log n / n) rather than sqrt(1 / n). That extra log is
    the cost of being allowed to peek forever, and it is unavoidable: the law of
    the iterated logarithm says no narrower boundary can hold at every n.
    """
    if n < 1:
        return math.inf
    ratio = (n + spec.n_opt) / spec.n_opt
    return spec.sigma * math.sqrt(
        (2.0 * (n + spec.n_opt) / (n * n)) * math.log(math.sqrt(ratio) / spec.alpha)
    )


@dataclass
class ConfidenceSequence:
    """Running mean with an interval that is valid at every sample size."""

    spec: ConfSeqSpec = ConfSeqSpec()
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def observe(self, value: float) -> None:
        self.n += 1
        self.total += value
        self.total_sq += value * value

    def observe_many(self, values: list[float]) -> None:
        for value in values:
            self.observe(value)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    @property
    def sample_variance(self) -> float:
        """Unbiased sample variance, reported for diagnostics only.

        The interval width does not use it: the Gaussian mixture boundary is
        driven by the assumed sub-Gaussian parameter, which is what makes the
        coverage guarantee hold without estimating the variance.
        """
        if self.n < 2:
            return 0.0
        mean = self.mean
        return max((self.total_sq - self.n * mean * mean) / (self.n - 1), 0.0)

    @property
    def radius(self) -> float:
        return radius(self.n, self.spec)

    def interval(self) -> tuple[float, float]:
        if self.n < 1:
            return (-math.inf, math.inf)
        r = self.radius
        return (self.mean - r, self.mean + r)

    def excludes(self, value: float) -> bool:
        """True once ``value`` has fallen outside the interval."""
        lo, hi = self.interval()
        return value < lo or value > hi

    def is_below(self, threshold: float) -> bool:
        """True once the whole interval sits below ``threshold``.

        This is the stopping rule for "the candidate's score dropped by more
        than the tolerance": run it on the paired differences with
        ``threshold = -tolerance``.
        """
        _, hi = self.interval()
        return hi < threshold

    def is_above(self, threshold: float) -> bool:
        lo, _ = self.interval()
        return lo > threshold

    def anytime_p_value(self, threshold: float = 0.0) -> float:
        """Smallest alpha at which the sequence would already exclude ``threshold``.

        The whole family of confidence sequences indexed by alpha is valid
        simultaneously, so inverting it gives a p-value that is itself valid at
        every sample size. This is what lets scalar-score tasks feed the same
        suite-level correction as pass/fail ones.

        Found by bisection because the radius is monotone in alpha but has no
        convenient closed-form inverse.
        """
        if self.n < 1:
            return 1.0
        distance = abs(self.mean - threshold)
        if distance <= 0.0:
            return 1.0

        def excludes_at(alpha: float) -> bool:
            spec = ConfSeqSpec(alpha=alpha, sigma=self.spec.sigma, n_opt=self.spec.n_opt)
            return radius(self.n, spec) < distance

        if not excludes_at(0.5):
            return 1.0
        lo, hi = 1e-12, 0.5
        for _ in range(200):
            mid = (lo * hi) ** 0.5  # geometric bisection: alpha spans many decades
            if excludes_at(mid):
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.0 + 1e-9:
                break
        return min(1.0, hi)

    def snapshot(self) -> dict[str, object]:
        lo, hi = self.interval()
        return {
            "n": self.n,
            "mean": self.mean,
            "lo": lo,
            "hi": hi,
            "radius": self.radius,
            "sample_variance": self.sample_variance,
        }
