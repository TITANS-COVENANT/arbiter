"""Persistence and explanation.

Storage matters here for one reason beyond durability: reusing yesterday's
baseline runs is usually a bigger saving than the sequential testing is. The
tests below pin the two properties that makes safe, namely that re-recording a
cell is idempotent and that changing the build changes its identity.
"""

from __future__ import annotations

import pytest

from arbiter.config import SuiteConfig, TargetConfig, TaskConfig
from arbiter.diff import (
    OpKind,
    align,
    cluster_failures,
    compare_modes,
    diff_task,
    normalise_error,
)
from arbiter.runner.types import RunOutcome, Step
from arbiter.store import Store


def outcome(passed: bool, error: str | None = None, names: tuple[str, ...] = ()) -> RunOutcome:
    steps = tuple(
        Step(index=i, kind="tool_call" if not n.startswith("!") else "error", name=n.lstrip("!"),
             ok=not n.startswith("!"))
        for i, n in enumerate(names)
    )
    return RunOutcome(passed=passed, error=error, steps=steps, cost_usd=0.01)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "runs.sqlite") as store:
        yield store


def suite(candidate_ref: str = "arbiter.sim.agent:candidate") -> SuiteConfig:
    return SuiteConfig(
        name="s",
        tasks=[TaskConfig(id="a"), TaskConfig(id="b")],
        baseline=TargetConfig(ref="arbiter.sim.agent:baseline"),
        candidate=TargetConfig(ref=candidate_ref),
    )


class TestStore:
    def test_round_trips_a_run(self, store):
        store.record_run(
            variant_id="v1", suite="s", task_id="a", replicate=0, seed=7,
            outcome=outcome(True, names=("plan", "answer")),
        )
        store.commit()
        loaded = store.load_runs("v1")
        assert loaded[("a", 0)].passed
        assert loaded[("a", 0)].steps[1].name == "answer"

    def test_recording_the_same_cell_twice_is_idempotent(self, store):
        for _ in range(3):
            store.record_run(
                variant_id="v1", suite="s", task_id="a", replicate=0, seed=7,
                outcome=outcome(True),
            )
        store.commit()
        assert store.count_runs("v1") == 1

    def test_variants_are_isolated(self, store):
        store.record_run(variant_id="v1", suite="s", task_id="a", replicate=0, seed=1,
                         outcome=outcome(True))
        store.record_run(variant_id="v2", suite="s", task_id="a", replicate=0, seed=1,
                         outcome=outcome(False))
        store.commit()
        assert store.load_runs("v1")[("a", 0)].passed
        assert not store.load_runs("v2")[("a", 0)].passed

    def test_filters_by_task(self, store):
        for task_id in ("a", "b"):
            store.record_run(variant_id="v1", suite="s", task_id=task_id, replicate=0,
                             seed=1, outcome=outcome(True))
        store.commit()
        assert set(store.load_runs("v1", ["a"])) == {("a", 0)}

    def test_task_runs_come_back_in_replicate_order(self, store):
        for replicate in (3, 1, 2, 0):
            store.record_run(variant_id="v1", suite="s", task_id="a", replicate=replicate,
                             seed=replicate, outcome=outcome(True))
        store.commit()
        assert [r for r, _ in store.load_task_runs("v1", "a")] == [0, 1, 2, 3]

    def test_pruning_removes_a_variant(self, store):
        store.record_run(variant_id="v1", suite="s", task_id="a", replicate=0, seed=1,
                         outcome=outcome(True))
        store.commit()
        assert store.prune_variant("v1") == 1
        assert store.count_runs("v1") == 0

    def test_gate_history_round_trips(self, store):
        summary = {
            "suite": "s", "baseline_variant": "b", "candidate_variant": "c",
            "verdict": "fail", "n_tasks": 2, "n_flagged": 1, "replicates_run": 40,
            "replicates_reused": 10, "cost_usd": 1.5, "wall_seconds": 3.0,
            "tasks": [{
                "task_id": "a", "verdict": "regression", "flagged": True, "replicates": 20,
                "e_value": 900.0, "anytime_p": 0.001, "adjusted_p": 0.002, "delta": -0.2,
            }],
        }
        gate_id = store.record_gate(summary)
        rows = store.recent_gates("s")
        assert len(rows) == 1
        assert rows[0]["verdict"] == "fail"
        assert store.gate_summary(gate_id)["n_flagged"] == 1

    def test_history_of_an_unknown_suite_is_empty(self, store):
        assert store.recent_gates("nope") == []


class TestVariantIdentity:
    def test_is_stable_across_calls(self):
        assert suite().variant_id("baseline") == suite().variant_id("baseline")

    def test_the_two_builds_differ(self):
        cfg = suite()
        assert cfg.variant_id("baseline") != cfg.variant_id("candidate")

    def test_changing_the_target_invalidates_the_cache(self):
        """The property that keeps baseline reuse honest."""
        assert suite().variant_id("candidate") != suite("other.module:fn").variant_id("candidate")

    def test_changing_the_seed_salt_invalidates_the_cache(self):
        a = suite()
        b = suite()
        b.seed_salt = "different"
        assert a.variant_id("baseline") != b.variant_id("baseline")

    def test_tightening_alpha_does_not_invalidate_the_cache(self):
        """Statistical policy is not part of what a stored run means."""
        a = suite()
        b = suite()
        b.stats.alpha = 0.001
        b.budget.max_cost_usd = 5.0
        assert a.variant_id("baseline") == b.variant_id("baseline")


class TestErrorNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("failed at row 4821", "failed at row <n>"),
            ("failed at row 991", "failed at row <n>"),
            ("timeout after 3.25 seconds", "timeout after <float> seconds"),
            ("bad id a3f9b2c1d4e5", "bad id <hex>"),
            ("cannot open /var/data/thing.json", "cannot open <path>"),
            ("fetch https://example.com/x failed", "fetch <url> failed"),
            ("expected 'alpha' got 'beta'", "expected <str> got <str>"),
        ],
    )
    def test_strips_the_volatile_parts(self, raw, expected):
        assert normalise_error(raw) == expected

    def test_two_instances_of_one_bug_collapse(self):
        assert normalise_error("row 12 missing") == normalise_error("row 8891 missing")

    def test_handles_missing_text(self):
        assert normalise_error(None) == "<no message>"
        assert normalise_error("   ") == "<no message>"


class TestClustering:
    def test_groups_by_normalised_message(self):
        outcomes = [
            outcome(False, "row 1 missing", ("!tool_error",)),
            outcome(False, "row 2 missing", ("!tool_error",)),
            outcome(False, "timeout", ("!timeout",)),
            outcome(True, None, ("answer",)),
        ]
        modes = cluster_failures(outcomes)
        assert len(modes) == 2
        assert modes[0].count == 2

    def test_separates_the_same_message_at_different_steps(self):
        outcomes = [
            outcome(False, "gave up", ("search", "!err")),
            outcome(False, "gave up", ("compute", "!other")),
        ]
        assert len(cluster_failures(outcomes)) == 2

    def test_ignores_passing_runs(self):
        assert cluster_failures([outcome(True), outcome(True)]) == []

    def test_compare_marks_a_new_mode(self):
        baseline = [outcome(False, "row 1 missing", ("!tool_error",))]
        candidate = [
            outcome(False, "row 2 missing", ("!tool_error",)),
            outcome(False, "agent looped", ("!loop",)),
        ]
        deltas = compare_modes(baseline, candidate)
        new = [d for d in deltas if d.is_new]
        assert len(new) == 1
        assert new[0].signature == "agent looped"

    def test_compare_marks_a_fixed_mode(self):
        deltas = compare_modes([outcome(False, "old bug", ("!e",))], [outcome(True)])
        assert deltas[0].is_fixed


class TestAlignment:
    def test_identical_trajectories_align_cleanly(self):
        steps = outcome(True, names=("plan", "search", "answer")).steps
        alignment = align(steps, steps)
        assert alignment.identical
        assert alignment.first_divergence is None
        assert alignment.similarity == 1.0

    def test_finds_where_a_run_broke(self):
        baseline = outcome(True, names=("plan", "search", "compute", "answer")).steps
        candidate = outcome(False, names=("plan", "search", "!tool_error")).steps
        alignment = align(baseline, candidate)
        assert not alignment.identical
        assert alignment.first_divergence == 2
        assert "tool_error" in " ".join(alignment.summary())

    def test_reports_an_extra_step_as_an_insert(self):
        baseline = outcome(True, names=("plan", "answer")).steps
        candidate = outcome(True, names=("plan", "search", "answer")).steps
        kinds = [op.kind for op in align(baseline, candidate).ops]
        assert OpKind.INSERT in kinds

    def test_reports_a_dropped_step_as_a_delete(self):
        baseline = outcome(True, names=("plan", "search", "answer")).steps
        candidate = outcome(True, names=("plan", "answer")).steps
        kinds = [op.kind for op in align(baseline, candidate).ops]
        assert OpKind.DELETE in kinds

    def test_same_tool_different_arguments_still_matches(self):
        """A different query is a match with changed arguments, not a rewrite."""
        left = (Step(index=0, kind="tool_call", name="search", args={"q": "a"}),)
        right = (Step(index=0, kind="tool_call", name="search", args={"q": "b"}),)
        alignment = align(left, right)
        assert alignment.ops[0].kind is OpKind.MATCH
        assert alignment.ops[0].args_changed
        assert alignment.first_divergence == 0

    def test_empty_trajectories(self):
        assert align((), ()).ops == ()


class TestTaskDiff:
    def test_picks_a_replicate_where_only_the_candidate_failed(self):
        baseline = [
            (0, outcome(False, "shared", ("!e",))),
            (1, outcome(True, names=("plan", "answer"))),
        ]
        candidate = [
            (0, outcome(False, "shared", ("!e",))),
            (1, outcome(False, "new bug", ("plan", "!loop"))),
        ]
        diff = diff_task("t", baseline, candidate)
        assert diff.representative_replicate == 1
        assert diff.baseline_passes == 1
        assert diff.candidate_passes == 0
        assert diff.alignment is not None
        assert not diff.alignment.identical

    def test_reports_new_modes(self):
        baseline = [(0, outcome(True))]
        candidate = [(0, outcome(False, "brand new", ("!x",)))]
        assert diff_task("t", baseline, candidate).new_modes[0].signature == "brand new"

    def test_survives_having_no_shared_replicates(self):
        diff = diff_task("t", [(0, outcome(True))], [(5, outcome(False, "x"))])
        assert diff.alignment is None
        assert diff.representative_replicate is None

    def test_rates_are_computed(self):
        baseline = [(i, outcome(i < 8)) for i in range(10)]
        candidate = [(i, outcome(i < 5)) for i in range(10)]
        diff = diff_task("t", baseline, candidate)
        assert diff.baseline_rate == pytest.approx(0.8)
        assert diff.candidate_rate == pytest.approx(0.5)
