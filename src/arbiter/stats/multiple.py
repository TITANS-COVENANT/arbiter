"""Controlling false discoveries across a whole suite.

Run 200 tasks at alpha=0.05 and about ten of them will flag a regression on a
build where nothing changed. Teams learn this the hard way, stop trusting the
gate, and start clicking merge anyway. Per-task error control is necessary and
nowhere near sufficient.

The fix is to control the false discovery rate over the suite: of the tasks the
gate flags, at most ``alpha`` of them should be flagged in error, on average.

Two procedures are provided. Benjamini-Hochberg on the anytime p-values is the
familiar one, but its guarantee needs the p-values to be independent or
positively dependent, and eval tasks that share a model, a prompt template and a
tool sandbox are dependent in ways nobody can characterise. e-BH takes e-values
instead and controls FDR under *arbitrary* dependence, with no assumption to
violate. Since the sequential tests in this package emit e-values natively,
e-BH is the default.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FdrResult", "benjamini_hochberg", "e_bh", "holm", "p_to_e"]


def p_to_e(p: float, kappa: float = 0.5) -> float:
    """Convert an anytime-valid p-value into an e-value.

    Uses the standard power calibrator ``e = kappa * p ** (kappa - 1)``. Under
    the null a valid p-value is stochastically larger than uniform, and
    ``E[kappa * U ** (kappa - 1)] = 1`` for uniform U, so the result satisfies
    the e-value property and can be mixed with genuine likelihood-ratio
    e-values in :func:`e_bh`.

    Some information is lost in the round trip, which is why the binary tests
    emit e-values directly rather than going through a p-value. This exists so
    that scalar-score tasks, whose evidence naturally arrives as a confidence
    sequence, can still take part in the same suite-level correction.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must lie in (0, 1), got {kappa}")
    p = min(max(p, 1e-300), 1.0)
    return kappa * p ** (kappa - 1.0)


@dataclass(frozen=True)
class FdrResult:
    """Which hypotheses survived correction, and the cutoff that did it."""

    rejected: list[int]
    adjusted: list[float]
    threshold: float
    procedure: str

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> FdrResult:
    """Benjamini-Hochberg step-up procedure.

    Returns the indices of rejected hypotheses along with BH-adjusted p-values
    (q-values), which are the numbers worth showing a human: an adjusted value
    of 0.02 means "flagging this and everything more extreme costs a 2% false
    discovery rate".
    """
    m = len(p_values)
    if m == 0:
        return FdrResult([], [], 0.0, "bh")
    order = sorted(range(m), key=lambda i: p_values[i])
    threshold = 0.0
    k = 0
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            k = rank
            threshold = alpha * rank / m
    rejected = sorted(order[:k])

    # Adjusted p-values come from a running minimum walked back from the largest
    # raw value, which enforces monotonicity.
    adjusted = [1.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return FdrResult(rejected, adjusted, threshold, "bh")


def e_bh(e_values: list[float], alpha: float = 0.05) -> FdrResult:
    """e-BH: FDR control under arbitrary dependence (Wang and Ramdas, 2022).

    Sort the e-values downward and reject the largest ``k`` where ``k`` is the
    biggest index whose e-value is at least ``m / (alpha * k)``. This is exactly
    Benjamini-Hochberg applied to ``p = 1 / e``, but because e-values only need
    ``E[e] <= 1`` under the null rather than a uniform distribution, the
    guarantee survives dependence between tasks. Given that every task in a
    suite shares a model and a prompt, that robustness is the whole reason to
    prefer it.
    """
    m = len(e_values)
    if m == 0:
        return FdrResult([], [], 0.0, "e-bh")
    order = sorted(range(m), key=lambda i: e_values[i], reverse=True)
    k = 0
    threshold = 0.0
    for rank, idx in enumerate(order, start=1):
        if e_values[idx] >= m / (alpha * rank):
            k = rank
            threshold = m / (alpha * rank)
    rejected = sorted(order[:k])
    adjusted = [min(1.0, (1.0 / e) if e > 0 else 1.0) for e in e_values]
    p_adjusted = benjamini_hochberg(adjusted, alpha).adjusted
    return FdrResult(rejected, p_adjusted, threshold, "e-bh")


def holm(p_values: list[float], alpha: float = 0.05) -> FdrResult:
    """Holm-Bonferroni step-down, controlling the family-wise error rate.

    Much stricter than FDR control: it bounds the probability of *any* false
    flag rather than their expected share. Worth switching to for a small,
    high-stakes suite where a single spurious block is expensive; far too
    conservative for a couple of hundred tasks.
    """
    m = len(p_values)
    if m == 0:
        return FdrResult([], [], 0.0, "holm")
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected: list[int] = []
    threshold = 0.0
    for rank, idx in enumerate(order, start=1):
        cutoff = alpha / (m - rank + 1)
        if p_values[idx] <= cutoff:
            rejected.append(idx)
            threshold = cutoff
        else:
            break
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        running = max(running, p_values[idx] * (m - rank + 1))
        adjusted[idx] = min(1.0, running)
    return FdrResult(sorted(rejected), adjusted, threshold, "holm")
