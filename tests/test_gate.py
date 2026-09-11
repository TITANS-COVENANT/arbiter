"""The gate end to end, plus scheduling and budgets.

These run against the simulated agent, so the truth is known and the assertions
can be about correctness rather than about not crashing.
"""

from __future__ import annotations

import pytest

from arbiter.config import (
    BudgetConfig,
    GateConfig,
    StatsConfig,
    SuiteConfig,
    TargetConfig,
    TaskConfig,
)
from arbiter.gate import StopReason, evidence_threshold, run_gate
from arbiter.gate.task_test import TaskTest
from arbiter.report import render_junit, render_markdown
from arbiter.runner.types import RunOutcome, Task
from arbiter.scheduler.allocator import (
    CheapestToCloseAllocator,
    RoundRobinAllocator,
    SuccessiveHalvingAllocator,
    build_allocator,
)
from arbiter.scheduler.budget import BudgetTracker
from arbiter.stats import Verdict
from arbiter.store import Store


def build_suite(
    *,
    clean: int = 4,
    regressed: int = 1,
    regressed_rate: float = 0.55,
    baseline_rate: float = 0.9,
    max_replicates: int = 120,
    correction: str = "e-bh",
    salt: str = "test",
    **budget_kwargs,
) -> SuiteConfig:
    tasks = [
        TaskConfig(
            id=f"bad_{i}",
            tags=["regressed"],
            input={
                "baseline_rate": baseline_rate,
                "candidate_rate": regressed_rate,
                "coupling": 0.7,
                "cost_usd": 0.01,
                "candidate_failure_mode": 2,
            },
        )
        for i in range(regressed)
    ] + [
        TaskConfig(
            id=f"ok_{i}",
            tags=["clean"],
            input={
                "baseline_rate": baseline_rate,
                "candidate_rate": baseline_rate,
                "coupling": 0.7,
                "cost_usd": 0.01,
            },
        )
        for i in range(clean)
    ]
    return SuiteConfig(
        name="test-suite",
        seed_salt=salt,
        tasks=tasks,
        baseline=TargetConfig(ref="arbiter.sim.agent:baseline", concurrency=32),
        candidate=TargetConfig(ref="arbiter.sim.agent:candidate", concurrency=32),
        stats=StatsConfig(
            max_replicates=max_replicates,
            min_replicates=4,
            correction=correction,  # type: ignore[arg-type]
        ),
        budget=BudgetConfig(batch_size=32, **budget_kwargs),
        gate=GateConfig(),
    )


class TestEvidenceThreshold:
    def test_scales_with_suite_size(self):
        assert evidence_threshold(200, 0.05, "e-bh") == 4000
        assert evidence_threshold(10, 0.05, "e-bh") == 200

    def test_uncorrected_is_just_ville(self):
        assert evidence_threshold(200, 0.05, "none") == 20


class TestGateEndToEnd:
    async def test_catches_a_clear_regression(self):
        result = await run_gate(build_suite(clean=4, regressed=1))
        assert result.verdict == "fail"
        assert result.exit_code == 1
        assert [t.task_id for t in result.flagged] == ["bad_0"]

    async def test_leaves_clean_tasks_alone(self):
        result = await run_gate(build_suite(clean=6, regressed=0))
        assert result.verdict in {"pass", "warn"}
        assert result.flagged == []
        assert result.exit_code == 0

    async def test_flagged_task_reports_the_direction_of_the_change(self):
        result = await run_gate(build_suite(clean=3, regressed=1))
        flagged = result.flagged[0]
        assert flagged.regressions > flagged.improvements
        assert flagged.delta < 0
        assert flagged.candidate_rate < flagged.baseline_rate

    async def test_every_task_stops_for_a_stated_reason(self):
        result = await run_gate(build_suite(clean=4, regressed=1))
        assert all(t.stop_reason is not StopReason.RUNNING for t in result.tasks)

    async def test_futility_stops_hopeless_tasks_early(self):
        """Clean tasks should not burn the whole per-task cap."""
        result = await run_gate(build_suite(clean=5, regressed=0, max_replicates=400))
        assert any(t.stop_reason is StopReason.FUTILITY for t in result.tasks)
        assert all(t.replicates < 400 for t in result.tasks)

    async def test_uncorrected_mode_flags_more_readily(self):
        corrected = await run_gate(
            build_suite(clean=8, regressed=1, regressed_rate=0.78, max_replicates=90)
        )
        uncorrected = await run_gate(
            build_suite(
                clean=8, regressed=1, regressed_rate=0.78, max_replicates=90, correction="none"
            )
        )
        assert len(uncorrected.flagged) >= len(corrected.flagged)


class TestBudgets:
    async def test_the_same_suite_produces_the_same_run(self):
        """Determinism, which is what makes a gate result arguable rather than magic."""
        first = await run_gate(build_suite(clean=6, regressed=1, max_replicates=200))
        second = await run_gate(build_suite(clean=6, regressed=1, max_replicates=200))
        assert first.replicates_run == second.replicates_run
        assert [t.task_id for t in first.flagged] == [t.task_id for t in second.flagged]
        assert [t.e_value for t in first.tasks] == [t.e_value for t in second.tasks]

    async def test_per_task_cap_is_never_exceeded(self):
        result = await run_gate(build_suite(clean=5, regressed=1, max_replicates=40))
        assert all(t.replicates <= 40 for t in result.tasks)

    async def test_a_truncated_run_says_so(self):
        suite = build_suite(clean=8, regressed=1, max_replicates=400)
        suite.budget.max_replicates = 200
        result = await run_gate(suite)
        assert result.binding_constraint == "max_replicates"
        assert result.incomplete
        assert result.verdict in {"warn", "fail"}
        assert "cut short" in " ".join(result.notes)

    async def test_a_truncated_run_can_be_configured_to_fail(self):
        suite = build_suite(clean=8, regressed=0, max_replicates=400)
        suite.budget.max_replicates = 100
        suite.gate.on_inconclusive = "fail"
        result = await run_gate(suite)
        assert result.verdict == "fail"
        assert result.exit_code == 1

    async def test_a_truncated_run_can_be_configured_to_pass(self):
        suite = build_suite(clean=6, regressed=0, max_replicates=400)
        suite.budget.max_replicates = 100
        suite.gate.on_inconclusive = "pass"
        result = await run_gate(suite)
        assert result.verdict == "pass"

    async def test_cost_ceiling_stops_the_run(self):
        suite = build_suite(clean=8, regressed=1, max_replicates=400)
        suite.budget.max_cost_usd = 0.5
        result = await run_gate(suite)
        assert result.cost_usd <= 1.5
        assert result.binding_constraint in {"max_cost_usd", None}


class TestBaselineReuse:
    async def test_second_run_reuses_the_baseline(self, tmp_path):
        """The saving that usually dwarfs the sequential testing."""
        suite = build_suite(clean=4, regressed=1)
        with Store(tmp_path / "runs.sqlite") as store:
            first = await run_gate(suite, store=store)
            assert first.replicates_reused == 0
            second = await run_gate(suite, store=store)
        assert second.replicates_reused > 0
        assert second.replicates_run < first.replicates_run
        assert second.reuse_rate > 0.5

    async def test_changing_the_candidate_invalidates_only_the_candidate(self, tmp_path):
        suite = build_suite(clean=3, regressed=1)
        with Store(tmp_path / "runs.sqlite") as store:
            await run_gate(suite, store=store)
            baseline_runs = store.count_runs(suite.variant_id("baseline"))
            suite.candidate.options = {"nudge": 1}
            second = await run_gate(suite, store=store)
            assert store.count_runs(suite.variant_id("baseline")) >= baseline_runs
        assert second.replicates_reused > 0

    async def test_verdicts_survive_a_restart(self, tmp_path):
        suite = build_suite(clean=3, regressed=1)
        with Store(tmp_path / "runs.sqlite") as store:
            result = await run_gate(suite, store=store)
            gate_id = store.record_gate(result.to_dict())
        with Store(tmp_path / "runs.sqlite") as store:
            assert store.gate_summary(gate_id)["verdict"] == result.verdict


class TestInfraErrors:
    async def test_an_unreachable_target_is_an_error_not_a_pass(self):
        suite = build_suite(clean=3, regressed=0, max_replicates=20)
        suite.candidate = TargetConfig(
            kind="python", ref=f"{__name__}:always_explodes", max_retries=0
        )
        result = await run_gate(suite)
        assert result.verdict == "error"
        assert result.exit_code == 2
        assert result.infra_error_rate > 0.25
        assert "not trustworthy" in " ".join(result.notes)

    async def test_errors_are_not_counted_as_failing_replicates(self):
        suite = build_suite(clean=2, regressed=0, max_replicates=20)
        suite.candidate = TargetConfig(
            kind="python", ref=f"{__name__}:always_explodes", max_retries=0
        )
        result = await run_gate(suite)
        assert all(t.replicates == 0 for t in result.tasks)
        assert all(t.infra_errors > 0 for t in result.tasks)


def always_explodes(task_input, seed, **_):
    raise RuntimeError("target is down")


class TestProgressAndReports:
    async def test_progress_callback_is_invoked(self):
        seen = []
        await run_gate(build_suite(clean=2, regressed=1), progress=seen.append)
        assert seen
        assert seen[-1]["total"] == 3

    async def test_markdown_report_names_the_flagged_task(self):
        result = await run_gate(build_suite(clean=3, regressed=1))
        markdown = render_markdown(result)
        assert "bad_0" in markdown
        assert "Regressions found" in markdown

    async def test_markdown_report_for_a_clean_build(self):
        result = await run_gate(build_suite(clean=3, regressed=0))
        markdown = render_markdown(result)
        assert "Nothing crossed the evidence threshold" in markdown

    async def test_junit_marks_flagged_tasks_as_failures(self):
        result = await run_gate(build_suite(clean=3, regressed=1))
        xml = render_junit(result)
        assert '<failure' in xml
        assert 'name="bad_0"' in xml
        assert 'tests="4"' in xml

    async def test_result_serialises_to_json_friendly_types(self):
        import json

        result = await run_gate(build_suite(clean=2, regressed=1))
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["n_tasks"] == 3
        assert isinstance(payload["tasks"], list)


class TestTaskTestUnit:
    @staticmethod
    def _test(**overrides) -> TaskTest:
        settings = {"min_replicates": 2, "max_replicates": 40, **overrides}
        return TaskTest(
            task=Task(id="t"), stats=StatsConfig(**settings), alpha_effective=0.05
        )

    def test_starts_running(self):
        assert self._test().stop_reason is StopReason.RUNNING
        assert not self._test().resolved

    def test_one_sided_failures_reach_the_boundary(self):
        test = self._test()
        for _ in range(40):
            test.observe(RunOutcome(passed=False), RunOutcome(passed=True))
            if test.resolved:
                break
        assert test.verdict is Verdict.REGRESSION
        assert test.stop_reason is StopReason.BOUNDARY

    def test_agreement_leads_to_futility(self):
        test = self._test()
        for _ in range(40):
            test.observe(RunOutcome(passed=True), RunOutcome(passed=True))
            if test.resolved:
                break
        assert test.stop_reason is StopReason.FUTILITY
        assert test.verdict is Verdict.INCONCLUSIVE

    def test_infra_errors_do_not_move_the_test(self):
        test = self._test()
        for _ in range(10):
            test.record_infra_error()
        assert test.replicates == 0
        assert test.e_value == 1.0
        assert test.result().infra_errors == 10

    def test_marking_capped_stops_a_task_with_no_data(self):
        test = self._test()
        test.mark_capped()
        assert test.resolved
        assert test.verdict is Verdict.INCONCLUSIVE

    def test_score_mode_uses_the_confidence_sequence(self):
        test = self._test(mode="score", score_tolerance=0.05, max_replicates=400)
        for _ in range(400):
            test.observe(RunOutcome(passed=True, score=0.1), RunOutcome(passed=True, score=0.9))
            if test.resolved:
                break
        assert test.verdict is Verdict.REGRESSION
        assert test.sequence is not None
        assert test.sequence.mean == pytest.approx(-0.8)

    def test_score_mode_clears_an_unchanged_task(self):
        test = self._test(mode="score", score_tolerance=0.3, max_replicates=800)
        for _ in range(800):
            test.observe(RunOutcome(passed=True, score=0.5), RunOutcome(passed=True, score=0.5))
            if test.resolved:
                break
        assert test.verdict is Verdict.PASS

    def test_score_mode_gives_no_evidence_when_the_change_is_upward(self):
        test = self._test(mode="score", score_tolerance=0.05, max_replicates=50)
        for _ in range(50):
            test.observe(RunOutcome(passed=True, score=0.9), RunOutcome(passed=True, score=0.1))
        assert test.e_value == 1.0
        assert test.anytime_p == 1.0


class _FakeState:
    def __init__(self, task_id, replicates=0, resolved=False, remaining=10.0, priority=0.0):
        self.task_id = task_id
        self._replicates = replicates
        self._resolved = resolved
        self._remaining = remaining
        self._priority = priority

    @property
    def replicates(self):
        return self._replicates

    @property
    def resolved(self):
        return self._resolved

    @property
    def priority(self):
        return self._priority

    def expected_remaining(self):
        return self._remaining


class TestAllocators:
    def test_nothing_is_scheduled_when_everything_resolved(self):
        states = [_FakeState("a", resolved=True), _FakeState("b", resolved=True)]
        for name in ("round-robin", "cheapest-to-close", "successive-halving"):
            assert build_allocator(name).select(states, 8) == []

    def test_round_robin_spreads_evenly(self):
        states = [_FakeState(x) for x in "abcd"]
        picks = RoundRobinAllocator().select(states, 8)
        assert len(picks) == 8
        assert sorted(set(picks)) == ["a", "b", "c", "d"]

    def test_round_robin_skips_resolved_tasks(self):
        states = [_FakeState("a"), _FakeState("b", resolved=True)]
        assert set(RoundRobinAllocator().select(states, 4)) == {"a"}

    def test_warmup_comes_before_greedy_ranking(self):
        states = [
            _FakeState("cheap", replicates=10, remaining=1.0),
            _FakeState("cold", replicates=0, remaining=999.0),
        ]
        picks = CheapestToCloseAllocator(min_replicates=4).select(states, 4)
        assert "cold" in picks

    def test_greedy_gives_near_tasks_exactly_what_they_need(self):
        """Two replicates from a decision means two replicates, not one."""
        states = [
            _FakeState("far", replicates=10, remaining=500.0),
            _FakeState("nearest", replicates=10, remaining=1.0),
            _FakeState("near", replicates=10, remaining=2.0),
        ]
        picks = CheapestToCloseAllocator(min_replicates=4, fairness=0.0).select(states, 4)
        assert picks.count("nearest") == 1
        assert picks.count("near") == 2
        assert picks.count("far") == 1

    def test_round_robin_ignores_how_close_a_task_is(self):
        states = [
            _FakeState("far", replicates=10, remaining=500.0),
            _FakeState("nearest", replicates=10, remaining=1.0),
            _FakeState("near", replicates=10, remaining=2.0),
        ]
        picks = RoundRobinAllocator().select(states, 4)
        assert picks.count("far") >= 1
        assert max(picks.count(t) for t in ("far", "near", "nearest")) <= 2

    def test_fairness_slice_still_samples_the_neglected(self):
        states = [
            _FakeState("hot", replicates=50, remaining=1.0),
            _FakeState("cold", replicates=6, remaining=900.0),
        ]
        picks = CheapestToCloseAllocator(min_replicates=4, fairness=0.5).select(states, 8)
        assert "cold" in picks

    def test_fairness_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="fairness"):
            CheapestToCloseAllocator(fairness=1.5)

    def test_successive_halving_concentrates_on_the_suspicious(self):
        states = [
            _FakeState("s0", replicates=8, priority=5.0),
            _FakeState("s1", replicates=8, priority=4.0),
            _FakeState("s2", replicates=8, priority=-1.0),
            _FakeState("s3", replicates=8, priority=-2.0),
        ]
        picks = set(SuccessiveHalvingAllocator(min_replicates=4).select(states, 8))
        assert "s0" in picks
        assert "s3" not in picks

    def test_keep_fraction_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="keep_fraction"):
            SuccessiveHalvingAllocator(keep_fraction=0.0)

    def test_unknown_name_falls_back_to_the_default(self):
        assert build_allocator("nonsense").name == "round-robin"

    def test_each_policy_is_reachable_by_name(self):
        for name in ("round-robin", "cheapest-to-close", "successive-halving"):
            assert build_allocator(name).name == name


class TestBudgetTracker:
    def test_starts_with_headroom(self):
        tracker = BudgetTracker(BudgetConfig(batch_size=16))
        assert not tracker.exhausted
        assert tracker.headroom() == 16

    def test_replicate_ceiling_binds(self):
        tracker = BudgetTracker(BudgetConfig(batch_size=16, max_replicates=20))
        tracker.charge(replicates=20)
        assert tracker.exhausted
        assert tracker.binding_constraint == "max_replicates"
        assert tracker.headroom() == 0

    def test_headroom_shrinks_near_the_ceiling(self):
        tracker = BudgetTracker(BudgetConfig(batch_size=16, max_replicates=20))
        tracker.charge(replicates=10)
        assert tracker.headroom() == 10

    def test_cost_ceiling_binds(self):
        tracker = BudgetTracker(BudgetConfig(batch_size=16, max_cost_usd=1.0))
        tracker.charge(replicates=10, cost_usd=1.0)
        assert tracker.binding_constraint == "max_cost_usd"

    def test_cost_headroom_uses_the_observed_rate(self):
        tracker = BudgetTracker(BudgetConfig(batch_size=100, max_cost_usd=1.0))
        tracker.charge(replicates=10, cost_usd=0.5)
        # $0.05 each with $0.50 left means ten more fit.
        assert tracker.headroom() == 10
