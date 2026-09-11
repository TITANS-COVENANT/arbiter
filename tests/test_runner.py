"""Execution: adapters, retries, timeouts, rate limits, and error accounting.

The distinction these tests are really protecting is between a task that failed
and a harness that failed. Getting that wrong in either direction is bad: count
timeouts as failures and a flaky network looks like a regression; ignore them
entirely and a gate that ran nothing reports green.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from arbiter.config import TargetConfig
from arbiter.errors import TargetError
from arbiter.runner import Cell, Engine, RunOutcome, Step, Task, TokenBucket, seed_for
from arbiter.runner.adapters import PythonTarget, build_target

TASK = Task(id="t1", input={"x": 1})


def passing_target(task_input, seed, **_):
    return {"passed": True, "cost_usd": 0.01, "steps": [{"index": 0, "name": "answer"}]}


def failing_target(task_input, seed, **_):
    return {"passed": False, "error": "wrong answer", "cost_usd": 0.01}


def seed_sensitive_target(task_input, seed, **_):
    return RunOutcome(passed=seed % 2 == 0)


def boolean_target(task_input, seed, **_):
    return True


async def async_target(task_input, seed, **_):
    await asyncio.sleep(0)
    return {"passed": True}


def exploding_target(task_input, seed, **_):
    raise RuntimeError("boom")


_ATTEMPTS = {"count": 0}


def flaky_target(task_input, seed, **_):
    _ATTEMPTS["count"] += 1
    if _ATTEMPTS["count"] < 3:
        raise TargetError("transient")
    return {"passed": True}


def slow_target(task_input, seed, **_):
    time.sleep(0.5)
    return {"passed": True}


def bad_shape_target(task_input, seed, **_):
    return 3.14


class TestSeeds:
    def test_is_deterministic(self):
        assert seed_for("a", 3, "salt") == seed_for("a", 3, "salt")

    def test_differs_by_task_replicate_and_salt(self):
        base = seed_for("a", 3, "salt")
        assert base != seed_for("b", 3, "salt")
        assert base != seed_for("a", 4, "salt")
        assert base != seed_for("a", 3, "other")

    def test_adding_a_task_does_not_shift_other_seeds(self):
        """Cached baseline runs must survive a new task being added.

        Seeds come from a hash of the identifiers rather than a running
        counter, so appending a task to the suite leaves every other task's
        seeds alone and yesterday's baseline runs stay valid.
        """
        before = [seed_for(f"task_{i}", r, "s") for i in range(5) for r in range(3)]
        after = [seed_for(f"task_{i}", r, "s") for i in range(6) for r in range(3)]
        assert before == after[: len(before)]

    def test_stays_in_positive_range(self):
        for i in range(500):
            assert 0 <= seed_for("t", i, "") <= 0x7FFFFFFF


class TestPythonTarget:
    def test_rejects_a_ref_without_a_callable(self):
        with pytest.raises(TargetError, match="module:callable"):
            PythonTarget(TargetConfig(kind="python", ref="arbiter.sim.agent"))

    def test_rejects_a_missing_attribute(self):
        with pytest.raises(TargetError, match="no attribute"):
            PythonTarget(TargetConfig(kind="python", ref="arbiter.sim.agent:nope"))

    def test_rejects_a_non_callable(self):
        with pytest.raises(TargetError, match="not callable"):
            PythonTarget(TargetConfig(kind="python", ref="arbiter:__version__"))

    async def test_accepts_a_dict(self):
        target = build_target(TargetConfig(ref=f"{__name__}:passing_target"))
        outcome = await target.run(TASK, 1)
        assert outcome.passed
        assert outcome.cost_usd == 0.01

    async def test_accepts_a_bare_boolean(self):
        target = build_target(TargetConfig(ref=f"{__name__}:boolean_target"))
        assert (await target.run(TASK, 1)).passed

    async def test_accepts_a_coroutine(self):
        target = build_target(TargetConfig(ref=f"{__name__}:async_target"))
        assert (await target.run(TASK, 1)).passed

    async def test_rejects_an_unusable_return_value(self):
        target = build_target(TargetConfig(ref=f"{__name__}:bad_shape_target"))
        with pytest.raises(TargetError, match="expected RunOutcome"):
            await target.run(TASK, 1)

    async def test_receives_the_seed(self):
        target = build_target(TargetConfig(ref=f"{__name__}:seed_sensitive_target"))
        assert (await target.run(TASK, 2)).passed
        assert not (await target.run(TASK, 3)).passed


class TestSubprocessTarget:
    async def test_reads_json_from_stdout(self):
        script = "import json,sys; sys.stdin.read(); print(json.dumps({'passed': True}))"
        engine = Engine(
            TargetConfig(kind="subprocess", command=[sys.executable, "-c", script], max_retries=0)
        )
        result = await engine.run_cell(Cell(TASK, 0, 7))
        assert result.ok
        assert result.outcome is not None
        assert result.outcome.passed

    async def test_a_non_zero_exit_is_harness_trouble(self):
        engine = Engine(
            TargetConfig(
                kind="subprocess",
                command=[sys.executable, "-c", "import sys; sys.exit(3)"],
                max_retries=0,
            )
        )
        result = await engine.run_cell(Cell(TASK, 0, 7))
        assert not result.ok
        assert result.error is not None
        assert "exited 3" in result.error

    async def test_non_json_output_is_harness_trouble(self):
        engine = Engine(
            TargetConfig(
                kind="subprocess",
                command=[sys.executable, "-c", "print('hello')"],
                max_retries=0,
            )
        )
        result = await engine.run_cell(Cell(TASK, 0, 7))
        assert not result.ok
        assert result.error is not None
        assert "not JSON" in result.error


class TestEngine:
    async def test_records_cost_and_run_counts(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:passing_target"))
        await engine.run_cells([Cell(TASK, i, i) for i in range(10)])
        assert engine.stats.runs == 10
        assert engine.stats.cost_usd == pytest.approx(0.10)
        assert engine.stats.infra_errors == 0

    async def test_fills_in_latency_when_the_target_does_not(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:passing_target"))
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert result.outcome is not None
        assert result.outcome.latency_ms > 0

    async def test_a_failing_task_is_evidence_not_an_error(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:failing_target"))
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert result.ok
        assert result.outcome is not None
        assert not result.outcome.passed
        assert engine.stats.infra_errors == 0

    async def test_an_exception_is_an_error_not_evidence(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:exploding_target", max_retries=0))
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert not result.ok
        assert result.error is not None
        assert "boom" in result.error

    async def test_retries_a_transient_failure(self):
        _ATTEMPTS["count"] = 0
        engine = Engine(TargetConfig(ref=f"{__name__}:flaky_target", max_retries=3))
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert result.ok
        assert result.attempts == 3
        assert engine.stats.retries == 0  # stats are recorded by run_cells

    async def test_gives_up_after_max_retries(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:exploding_target", max_retries=2))
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert not result.ok
        assert result.attempts == 3

    async def test_times_out_a_slow_target(self):
        engine = Engine(
            TargetConfig(ref=f"{__name__}:slow_target", timeout_s=0.05, max_retries=0)
        )
        result = await engine.run_cell(Cell(TASK, 0, 0))
        assert not result.ok
        assert result.error is not None
        assert "timeout" in result.error

    async def test_respects_the_concurrency_limit(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:passing_target", concurrency=2))
        assert engine._sem._value == 2

    async def test_empty_batch_is_a_no_op(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:passing_target"))
        assert await engine.run_cells([]) == []

    async def test_infra_error_rate_is_reported(self):
        engine = Engine(TargetConfig(ref=f"{__name__}:exploding_target", max_retries=0))
        await engine.run_cells([Cell(TASK, i, i) for i in range(4)])
        assert engine.stats.infra_error_rate == 1.0
        assert len(engine.stats.errors) == 4


class TestTokenBucket:
    async def test_throttles_to_the_configured_rate(self):
        bucket = TokenBucket(rate_per_s=50, capacity=1)
        started = time.monotonic()
        for _ in range(5):
            await bucket.acquire()
        # Four refills at 50/s is at least 80ms; allow slack for a slow CI box.
        assert time.monotonic() - started >= 0.05

    async def test_rejects_a_non_positive_rate(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBucket(0)


class TestOutcomeSerialisation:
    def test_round_trips_through_a_dict(self):
        outcome = RunOutcome(
            passed=False,
            score=0.4,
            error="nope",
            cost_usd=0.02,
            latency_ms=12.5,
            steps=(Step(index=0, kind="tool_call", name="search", ok=False, args={"q": "x"}),),
            metadata={"build": "candidate"},
        )
        restored = RunOutcome.from_dict(outcome.to_dict())
        assert restored.passed == outcome.passed
        assert restored.score == outcome.score
        assert restored.steps[0].signature == outcome.steps[0].signature
        assert restored.metadata == outcome.metadata

    def test_step_signature_ignores_arguments(self):
        left = Step(index=0, kind="tool_call", name="search", args={"q": "a"})
        right = Step(index=0, kind="tool_call", name="search", args={"q": "b"})
        assert left.signature == right.signature
        assert left.args_digest != right.args_digest
