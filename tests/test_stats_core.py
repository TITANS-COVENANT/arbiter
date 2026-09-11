"""Numerics and the sequential tests.

The Monte Carlo tests here are the ones that matter. Everything else in this
repo is plumbing around the claim that the false-positive rate is bounded, and a
unit test that only checks the code runs would not notice if that claim stopped
being true.
"""

from __future__ import annotations

import math
import random

import pytest

from arbiter.stats import (
    BetaPosterior,
    ConfidenceSequence,
    ConfSeqSpec,
    PairedSpec,
    PairedSprt,
    PairOutcome,
    Sprt,
    SprtSpec,
    Verdict,
    beta_ppf,
    betainc,
    fixed_sample_size,
    norm_cdf,
    norm_ppf,
    paired_plan,
    plan_suite,
    prob_worse,
    radius,
)


class TestSpecialFunctions:
    @pytest.mark.parametrize(
        ("a", "b", "x", "expected"),
        [
            (2.0, 3.0, 0.5, 0.6875),
            (0.5, 0.5, 0.25, 1.0 / 3.0),
            (5.0, 1.0, 0.5, 0.03125),
            (1.0, 1.0, 0.42, 0.42),
            (10.0, 2.0, 0.9, 0.69735688),  # x^a * (a + 1 - a x) for b = 2
        ],
    )
    def test_betainc_matches_known_values(self, a, b, x, expected):
        assert betainc(a, b, x) == pytest.approx(expected, abs=1e-9)

    def test_betainc_is_a_cdf(self):
        assert betainc(3, 4, 0.0) == 0.0
        assert betainc(3, 4, 1.0) == 1.0
        previous = 0.0
        for i in range(1, 100):
            value = betainc(3, 4, i / 100)
            assert value >= previous
            previous = value

    def test_betainc_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="positive shape"):
            betainc(0.0, 1.0, 0.5)

    def test_beta_ppf_inverts_betainc(self):
        for a, b in [(2, 3), (10, 2), (0.5, 0.5), (7, 7)]:
            for q in (0.05, 0.25, 0.5, 0.9, 0.99):
                assert betainc(a, b, beta_ppf(a, b, q)) == pytest.approx(q, abs=1e-8)

    @pytest.mark.parametrize(
        ("p", "expected"),
        [(0.975, 1.959964), (0.95, 1.644854), (0.5, 0.0), (0.001, -3.090232)],
    )
    def test_norm_ppf_matches_known_quantiles(self, p, expected):
        assert norm_ppf(p) == pytest.approx(expected, abs=1e-6)

    def test_norm_ppf_inverts_norm_cdf(self):
        for i in range(1, 1000):
            p = i / 1000
            assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-12)

    def test_norm_ppf_rejects_endpoints(self):
        with pytest.raises(ValueError, match="0 < p < 1"):
            norm_ppf(0.0)


class TestSprtConstruction:
    def test_from_mde_sets_the_alternative(self):
        spec = SprtSpec.from_mde(0.9, 0.15)
        assert spec.p0 == pytest.approx(0.9)
        assert spec.p1 == pytest.approx(0.75)

    def test_rejects_inverted_hypotheses(self):
        with pytest.raises(ValueError, match="p1 < p0"):
            SprtSpec(p0=0.5, p1=0.8)

    def test_boundaries_have_the_right_signs(self):
        spec = SprtSpec.from_mde(0.9, 0.15)
        assert spec.upper > 0 > spec.lower
        assert spec.step_pass < 0 < spec.step_fail

    def test_failures_push_toward_the_regression_boundary(self):
        test = Sprt(SprtSpec.from_mde(0.9, 0.15))
        for _ in range(20):
            test.observe(False)
        assert test.verdict is Verdict.REGRESSION

    def test_passes_push_toward_acquittal(self):
        test = Sprt(SprtSpec.from_mde(0.9, 0.15))
        for _ in range(40):
            test.observe(True)
        assert test.verdict is Verdict.PASS

    def test_min_samples_defers_the_decision(self):
        test = Sprt(SprtSpec.from_mde(0.9, 0.15, min_samples=30))
        for _ in range(20):
            test.observe(False)
        assert test.verdict is Verdict.CONTINUE

    def test_truncation_reports_inconclusive(self):
        spec = SprtSpec(p0=0.9, p1=0.88, max_samples=5)
        test = Sprt(spec)
        test.observe_many([True, False, True, False, True])
        assert test.verdict is Verdict.INCONCLUSIVE

    def test_observe_many_stops_at_the_boundary(self):
        test = Sprt(SprtSpec.from_mde(0.9, 0.15))
        test.observe_many([False] * 100)
        assert test.n < 100

    def test_operating_characteristic_brackets_the_error_budget(self):
        spec = SprtSpec.from_mde(0.9, 0.15, alpha=0.05, beta=0.10, max_samples=500)
        accept_h0_at_null, _ = spec.operating_characteristic(spec.p0)
        accept_h0_at_alt, _ = spec.operating_characteristic(spec.p1)
        assert accept_h0_at_null == pytest.approx(1 - spec.alpha, abs=0.02)
        assert accept_h0_at_alt == pytest.approx(spec.beta, abs=0.02)

    def test_expected_remaining_falls_to_zero_once_decided(self):
        test = Sprt(SprtSpec.from_mde(0.9, 0.15))
        assert test.expected_remaining() > 0
        test.observe_many([False] * 50)
        assert test.expected_remaining() == 0.0


class TestSprtErrorRates:
    """The guarantees, checked by simulation rather than by assertion."""

    @staticmethod
    def _run(spec: SprtSpec, p: float, trials: int, seed: int):
        rng = random.Random(seed)
        outcomes = {Verdict.PASS: 0, Verdict.REGRESSION: 0, Verdict.INCONCLUSIVE: 0}
        total_n = 0
        for _ in range(trials):
            test = Sprt(spec)
            while not test.verdict.is_terminal:
                test.observe(rng.random() < p)
            outcomes[test.verdict] += 1
            total_n += test.n
        return outcomes, total_n / trials

    def test_false_positive_rate_stays_under_alpha(self):
        spec = SprtSpec.from_mde(0.9, 0.15, alpha=0.05, beta=0.10, max_samples=500)
        outcomes, _ = self._run(spec, spec.p0, trials=6000, seed=11)
        false_positive_rate = outcomes[Verdict.REGRESSION] / 6000
        assert false_positive_rate <= 0.05 + 0.012

    def test_miss_rate_stays_under_beta(self):
        spec = SprtSpec.from_mde(0.9, 0.15, alpha=0.05, beta=0.10, max_samples=500)
        outcomes, _ = self._run(spec, spec.p1, trials=6000, seed=12)
        miss_rate = outcomes[Verdict.PASS] / 6000
        assert miss_rate <= 0.10 + 0.015

    def test_costs_less_than_the_fixed_sample_equivalent(self):
        spec = SprtSpec.from_mde(0.9, 0.15, alpha=0.05, beta=0.10, max_samples=500)
        _, mean_n = self._run(spec, spec.p0, trials=3000, seed=13)
        fixed = fixed_sample_size(spec.p0, spec.p1, spec.alpha, spec.beta)
        assert mean_n < fixed * 0.8

    def test_e_value_expectation_is_bounded_by_one_under_the_null(self):
        """The property e-BH depends on, checked directly.

        A supermartingale that starts at 1 and is stopped at a stopping time
        still has expectation at most 1. If this ever fails, the suite-level FDR
        guarantee is void.
        """
        spec = SprtSpec.from_mde(0.9, 0.15, max_samples=200)
        rng = random.Random(99)
        total = 0.0
        trials = 4000
        for _ in range(trials):
            test = Sprt(spec)
            while not test.verdict.is_terminal:
                test.observe(rng.random() < spec.p0)
            total += test.e_value
        assert total / trials <= 1.15

    def test_anytime_p_value_is_uniform_or_larger_under_the_null(self):
        spec = SprtSpec.from_mde(0.9, 0.15, max_samples=200)
        rng = random.Random(7)
        trials = 3000
        for level in (0.05, 0.10, 0.20):
            below = 0
            for _ in range(trials):
                test = Sprt(spec)
                while not test.verdict.is_terminal:
                    test.observe(rng.random() < spec.p0)
                below += test.anytime_p_value <= level
            assert below / trials <= level + 0.02


class TestPairedSprt:
    def test_concordant_pairs_carry_no_information(self):
        test = PairedSprt(PairedSpec())
        for _ in range(50):
            test.observe(PairOutcome(candidate_passed=True, baseline_passed=True))
            test.observe(PairOutcome(candidate_passed=False, baseline_passed=False))
        assert test.discordant == 0
        assert test.inner.log_lr == 0.0
        assert test.replicates == 100

    def test_one_sided_disagreement_flags_a_regression(self):
        test = PairedSprt(PairedSpec(max_replicates=500))
        for _ in range(60):
            test.observe(PairOutcome(candidate_passed=False, baseline_passed=True))
            if test.verdict.is_terminal:
                break
        assert test.verdict is Verdict.REGRESSION
        assert test.regressions > 0

    def test_symmetric_disagreement_acquits(self):
        test = PairedSprt(PairedSpec(max_replicates=2000))
        for _ in range(200):
            test.observe(PairOutcome(candidate_passed=False, baseline_passed=True))
            test.observe(PairOutcome(candidate_passed=True, baseline_passed=False))
            if test.verdict.is_terminal:
                break
        assert test.verdict is Verdict.PASS

    def test_delta_tracks_the_pass_rate_change(self):
        test = PairedSprt(PairedSpec(max_replicates=100))
        for _ in range(10):
            test.observe(PairOutcome(candidate_passed=False, baseline_passed=True))
        for _ in range(90):
            test.observe(PairOutcome(candidate_passed=True, baseline_passed=True))
        assert test.delta == pytest.approx(-0.10)
        assert test.baseline_rate == pytest.approx(1.0)
        assert test.candidate_rate == pytest.approx(0.90)

    def test_mcnemar_p_value_is_symmetric(self):
        left = PairedSprt(PairedSpec(max_replicates=100))
        right = PairedSprt(PairedSpec(max_replicates=100))
        for _ in range(8):
            left.observe(PairOutcome(candidate_passed=False, baseline_passed=True))
            right.observe(PairOutcome(candidate_passed=True, baseline_passed=False))
        assert left.mcnemar_p_value() == pytest.approx(right.mcnemar_p_value())

    def test_false_positive_rate_under_a_true_null(self):
        """Two identical builds, differing only by independent noise."""
        spec = PairedSpec(alpha=0.05, beta=0.10, max_discordant=400, max_replicates=4000)
        rng = random.Random(4242)
        trials = 3000
        flagged = 0
        for _ in range(trials):
            test = PairedSprt(spec)
            while not test.verdict.is_terminal:
                # Under the null a disagreement is a fair coin.
                if rng.random() < 0.2:
                    regression = rng.random() < 0.5
                    test.observe(
                        PairOutcome(
                            candidate_passed=not regression, baseline_passed=regression
                        )
                    )
                else:
                    test.observe(PairOutcome(candidate_passed=True, baseline_passed=True))
            flagged += test.verdict is Verdict.REGRESSION
        assert flagged / trials <= 0.05 + 0.012

    def test_expected_remaining_accounts_for_the_disagreement_rate(self):
        """A task whose builds rarely disagree is expensive, and must say so."""
        chatty = PairedSprt(PairedSpec(max_replicates=10000))
        quiet = PairedSprt(PairedSpec(max_replicates=10000))
        for i in range(100):
            chatty.observe(
                PairOutcome(candidate_passed=i % 2 == 0, baseline_passed=i % 2 == 1)
            )
            quiet.observe(PairOutcome(candidate_passed=True, baseline_passed=True))
        assert quiet.expected_remaining() > chatty.expected_remaining()


class TestConfidenceSequence:
    def test_interval_contains_the_mean(self):
        sequence = ConfidenceSequence(ConfSeqSpec(alpha=0.05, sigma=0.5))
        sequence.observe_many([0.5] * 50)
        low, high = sequence.interval()
        assert low < 0.5 < high

    def test_radius_shrinks_with_more_data(self):
        spec = ConfSeqSpec(alpha=0.05, sigma=0.5, n_opt=50)
        assert radius(10, spec) > radius(100, spec) > radius(1000, spec)

    def test_radius_is_wider_than_a_fixed_sample_interval(self):
        """The price of being allowed to peek."""
        spec = ConfSeqSpec(alpha=0.05, sigma=0.5, n_opt=50)
        fixed = 1.959964 * 0.5 / math.sqrt(50)
        assert radius(50, spec) > fixed

    def test_coverage_holds_uniformly_over_time(self):
        """The guarantee: the true mean never escapes, more than alpha of the time."""
        spec = ConfSeqSpec(alpha=0.05, sigma=0.5, n_opt=30)
        rng = random.Random(2024)
        trials, horizon, escapes = 800, 200, 0
        true_mean = 0.4
        for _ in range(trials):
            sequence = ConfidenceSequence(spec)
            for _ in range(horizon):
                sequence.observe(1.0 if rng.random() < true_mean else 0.0)
                if sequence.excludes(true_mean):
                    escapes += 1
                    break
        assert escapes / trials <= 0.05

    def test_is_below_detects_a_threshold_crossing(self):
        sequence = ConfidenceSequence(ConfSeqSpec(alpha=0.05, sigma=1.0, n_opt=20))
        sequence.observe_many([-0.9] * 200)
        assert sequence.is_below(-0.05)
        assert not sequence.is_above(-0.05)

    def test_anytime_p_value_falls_as_evidence_accumulates(self):
        sequence = ConfidenceSequence(ConfSeqSpec(alpha=0.05, sigma=1.0, n_opt=20))
        previous = 1.0
        for _ in range(6):
            sequence.observe_many([-0.8] * 20)
            current = sequence.anytime_p_value(0.0)
            assert current <= previous
            previous = current
        assert previous < 0.01

    def test_anytime_p_value_stays_at_one_without_evidence(self):
        sequence = ConfidenceSequence(ConfSeqSpec(alpha=0.05, sigma=1.0))
        sequence.observe_many([0.0] * 50)
        assert sequence.anytime_p_value(0.0) == 1.0


class TestBayes:
    def test_posterior_updates_from_counts(self):
        posterior = BetaPosterior.from_data(9, 10)
        assert posterior.alpha == 10
        assert posterior.beta == 2
        assert posterior.mean == pytest.approx(10 / 12)

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError, match="0 <= passes <= n"):
            BetaPosterior.from_data(11, 10)

    def test_credible_interval_brackets_the_mean(self):
        posterior = BetaPosterior.from_data(45, 50)
        low, high = posterior.credible_interval(0.95)
        assert low < posterior.mean < high
        assert high - low < 0.25

    def test_identical_data_gives_even_odds(self):
        assert prob_worse(18, 20, 18, 20) == pytest.approx(0.5, abs=1e-6)

    def test_matches_monte_carlo(self):
        rng = random.Random(5)
        for base_passes, base_n, cand_passes, cand_n, delta in [
            (9, 10, 6, 10, 0.0),
            (45, 50, 38, 50, 0.05),
            (90, 100, 80, 100, 0.05),
        ]:
            analytic = prob_worse(base_passes, base_n, cand_passes, cand_n, delta)
            hits = 0
            trials = 40000
            for _ in range(trials):
                p_base = rng.betavariate(1 + base_passes, 1 + base_n - base_passes)
                p_cand = rng.betavariate(1 + cand_passes, 1 + cand_n - cand_passes)
                hits += p_cand < p_base - delta
            assert analytic == pytest.approx(hits / trials, abs=0.01)

    def test_tolerance_reduces_the_probability(self):
        without = prob_worse(90, 100, 80, 100, 0.0)
        with_tolerance = prob_worse(90, 100, 80, 100, 0.08)
        assert with_tolerance < without


class TestPlanning:
    def test_fixed_sample_size_grows_as_the_effect_shrinks(self):
        assert fixed_sample_size(0.9, 0.6) < fixed_sample_size(0.9, 0.75)
        assert fixed_sample_size(0.9, 0.75) < fixed_sample_size(0.9, 0.85)

    def test_fixed_sample_size_grows_as_alpha_tightens(self):
        assert fixed_sample_size(0.9, 0.75, 0.05) < fixed_sample_size(0.9, 0.75, 0.001)

    def test_sequential_plan_beats_fixed_sample(self):
        plan = plan_suite(n_tasks=50, baseline_rate=0.9, mde=0.15)
        assert plan.expected_replicates < plan.fixed_replicates
        assert 0.0 < plan.savings < 1.0

    def test_plan_scales_cost_and_time(self):
        plan = plan_suite(
            n_tasks=10,
            baseline_rate=0.9,
            mde=0.15,
            cost_per_replicate=0.5,
            seconds_per_replicate=4.0,
            concurrency=4,
        )
        assert plan.fixed_cost == pytest.approx(plan.fixed_replicates * 0.5)
        assert plan.fixed_wall_clock_seconds == pytest.approx(plan.fixed_replicates * 4.0 / 4)

    def test_paired_plan_needs_more_replicates_for_a_bigger_suite(self):
        small = paired_plan(n_tasks=10, baseline_rate=0.9, mde=0.15)
        large = paired_plan(n_tasks=500, baseline_rate=0.9, mde=0.15)
        assert large.replicates_needed > small.replicates_needed
        assert large.evidence_threshold > small.evidence_threshold

    def test_paired_plan_rewards_tighter_coupling(self):
        """Seeds that actually control the target buy real savings."""
        loose = paired_plan(n_tasks=50, baseline_rate=0.9, mde=0.15, coupling=0.1)
        tight = paired_plan(n_tasks=50, baseline_rate=0.9, mde=0.15, coupling=0.9)
        assert tight.discordance_rate > loose.discordance_rate * 0.5
        assert tight.replicates_needed < loose.replicates_needed

    def test_paired_plan_without_correction_is_cheaper(self):
        corrected = paired_plan(n_tasks=100, baseline_rate=0.9, mde=0.15)
        uncorrected = paired_plan(n_tasks=100, baseline_rate=0.9, mde=0.15, correction="none")
        assert uncorrected.replicates_needed < corrected.replicates_needed
