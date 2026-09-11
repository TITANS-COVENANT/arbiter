"""Suite-level correction.

The headline test is :meth:`TestFdrControl.test_e_bh_controls_fdr_under_dependence`.
Eval tasks in one suite share a model, a prompt template and a sandbox, so their
evidence is dependent in ways nobody can characterise, and a procedure that
needs independence is a procedure whose guarantee does not apply.
"""

from __future__ import annotations

import random
import statistics

import pytest

from arbiter.stats import benjamini_hochberg, e_bh, holm, p_to_e
from arbiter.stats.sprt import Sprt, SprtSpec


class TestBenjaminiHochberg:
    def test_empty_input(self):
        result = benjamini_hochberg([], 0.05)
        assert result.rejected == []
        assert result.n_rejected == 0

    def test_rejects_the_obvious_ones(self):
        result = benjamini_hochberg([0.001, 0.01, 0.04, 0.9], 0.05)
        assert result.rejected == [0, 1]

    def test_rejects_nothing_when_everything_is_null(self):
        assert benjamini_hochberg([0.4, 0.6, 0.8, 0.95], 0.05).rejected == []

    def test_step_up_rescues_borderline_values(self):
        """The point of BH: many mediocre p-values together clear the bar."""
        p_values = [0.01] * 10
        assert len(benjamini_hochberg(p_values, 0.05).rejected) == 10
        assert benjamini_hochberg([0.01, 0.9, 0.9, 0.9, 0.9], 0.05).rejected == [0]

    def test_adjusted_values_are_monotone(self):
        result = benjamini_hochberg([0.001, 0.01, 0.02, 0.3, 0.7], 0.05)
        ordered = sorted(zip(result.adjusted, [0.001, 0.01, 0.02, 0.3, 0.7], strict=True))
        adjusted = [a for a, _ in ordered]
        assert adjusted == sorted(adjusted)

    def test_adjusted_values_never_shrink_below_raw(self):
        raw = [0.001, 0.02, 0.3]
        result = benjamini_hochberg(raw, 0.05)
        assert all(a >= r - 1e-12 for a, r in zip(result.adjusted, raw, strict=True))


class TestEBh:
    def test_empty_input(self):
        assert e_bh([], 0.05).rejected == []

    def test_single_strong_e_value(self):
        # With m=4 and alpha=0.05 the first rung demands e >= 80.
        assert e_bh([100.0, 1.0, 1.0, 1.0], 0.05).rejected == [0]
        assert e_bh([70.0, 1.0, 1.0, 1.0], 0.05).rejected == []

    def test_step_up_lowers_the_bar_for_a_group(self):
        """Four tasks at e=40 each: none clears rung one, all clear rung four."""
        assert e_bh([40.0] * 4, 0.05).rejected == [0, 1, 2, 3]

    def test_equivalent_to_bh_on_reciprocals(self):
        e_values = [100.0, 45.0, 3.0, 1.2, 0.9]
        by_e = set(e_bh(e_values, 0.05).rejected)
        by_p = set(benjamini_hochberg([1 / e for e in e_values], 0.05).rejected)
        assert by_e == by_p

    def test_handles_zero_and_tiny_e_values(self):
        assert e_bh([0.0, 0.0, 1e-12], 0.05).rejected == []


class TestHolm:
    def test_is_stricter_than_bh(self):
        p_values = [0.001, 0.01, 0.02, 0.04, 0.049]
        assert len(holm(p_values, 0.05).rejected) <= len(
            benjamini_hochberg(p_values, 0.05).rejected
        )

    def test_stops_at_the_first_failure(self):
        # Sorted: 0.001 clears alpha/3, then 0.03 misses alpha/2 and the
        # step-down halts, so 0.9 is never considered.
        assert holm([0.001, 0.9, 0.03], 0.05).rejected == [0]

    def test_steps_down_in_sorted_order_not_input_order(self):
        # Both small values clear their rung; position in the input is
        # irrelevant, which is the part that trips people up when reading
        # results.
        assert holm([0.001, 0.9, 0.001], 0.05).rejected == [0, 2]

    def test_empty_input(self):
        assert holm([], 0.05).rejected == []


class TestCalibrator:
    def test_small_p_becomes_large_e(self):
        assert p_to_e(0.001) > p_to_e(0.01) > p_to_e(0.5)

    def test_expectation_under_the_null_is_one(self):
        """A valid calibrator maps uniform p-values to mean-one e-values."""
        rng = random.Random(3)
        values = [p_to_e(rng.random()) for _ in range(200000)]
        assert statistics.fmean(values) == pytest.approx(1.0, abs=0.05)

    def test_rejects_bad_kappa(self):
        with pytest.raises(ValueError, match="kappa"):
            p_to_e(0.1, kappa=1.5)


class TestFdrControl:
    """Realised false discovery rate, measured rather than asserted."""

    @staticmethod
    def _suite_e_values(
        rng: random.Random, n_tasks: int, n_regressed: int, shared_shock: float
    ) -> list[float]:
        """Run one suite of sequential tests, with a shock shared by every task.

        The shared shock is what makes the tasks dependent. A bad day for the
        model is a bad day for every task at once, which is exactly the
        structure BH's independence assumption does not cover and e-BH's
        guarantee does.
        """
        spec = SprtSpec.from_mde(0.9, 0.15, alpha=0.05, beta=0.10, max_samples=120)
        shock = rng.gauss(0.0, shared_shock)
        e_values = []
        for index in range(n_tasks):
            rate = spec.p1 if index < n_regressed else spec.p0
            rate = min(max(rate + shock, 0.05), 0.99)
            test = Sprt(spec)
            while not test.verdict.is_terminal:
                test.observe(rng.random() < rate)
            e_values.append(test.e_value)
        return e_values

    def test_e_bh_controls_fdr_under_dependence(self):
        rng = random.Random(20260910)
        n_tasks, n_regressed, alpha = 20, 4, 0.10
        proportions = []
        for _ in range(400):
            e_values = self._suite_e_values(rng, n_tasks, n_regressed, shared_shock=0.04)
            rejected = e_bh(e_values, alpha).rejected
            false_positives = sum(1 for i in rejected if i >= n_regressed)
            proportions.append(false_positives / len(rejected) if rejected else 0.0)
        assert statistics.fmean(proportions) <= alpha

    def test_uncorrected_testing_is_visibly_worse(self):
        """Why the correction is not optional.

        With no correction, a suite of twenty clean tasks flags something on a
        large share of runs. That is the behaviour that trains people to ignore
        the gate.
        """
        rng = random.Random(555)
        alpha = 0.05
        uncorrected_noisy_runs = 0
        corrected_noisy_runs = 0
        trials = 300
        for _ in range(trials):
            e_values = self._suite_e_values(rng, n_tasks=20, n_regressed=0, shared_shock=0.0)
            uncorrected_noisy_runs += any(e >= 1 / alpha for e in e_values)
            corrected_noisy_runs += bool(e_bh(e_values, alpha).rejected)
        assert corrected_noisy_runs < uncorrected_noisy_runs
        assert corrected_noisy_runs / trials <= alpha
