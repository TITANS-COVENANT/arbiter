"""A synthetic agent with a dial for every property that makes evals hard.

Claims about false-positive rates and sample savings are worth nothing unless
someone can check them, and you cannot check them against a real agent because
you never know its true pass rate. So arbiter ships a fake one whose truth is
known by construction, and the benchmarks measure the gate against it.

Three properties are modelled, and they are the three that matter:

**Stochasticity.** A task has a pass probability, not a pass/fail answer. Run it
twice and you may get two answers.

**Coupling.** Fixing the seed removes some of the randomness but not all of it.
A shared latent draw decides the outcome on a ``coupling`` fraction of
replicates and the two builds diverge independently on the rest. Set coupling to
1 and the builds are deterministic twins; set it to 0 and seeding buys nothing.
Real agents sit in between, which is exactly the regime where paired testing
earns its keep and where a naive unpaired comparison wastes money.

**Failure modes.** Failures are not interchangeable. A regression usually shows
up as one specific new way of failing, which is what the trajectory diff is for.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..runner.types import RunOutcome, Step

__all__ = ["SimSpec", "baseline", "candidate", "simulate_run"]

_LATENT_SALT = 0x5EED_1CE
_FAILURE_MODES = (
    ("tool_error", "search returned no results"),
    ("wrong_answer", "final answer did not match expected value"),
    ("loop", "agent repeated the same tool call and ran out of steps"),
    ("refusal", "agent declined to answer"),
    ("timeout", "agent exceeded the step budget"),
)


@dataclass(frozen=True)
class SimSpec:
    """Ground truth for one simulated task."""

    baseline_rate: float = 0.9
    candidate_rate: float = 0.9
    coupling: float = 0.7
    cost_usd: float = 0.02
    latency_ms: float = 900.0
    steps: int = 4
    failure_mode: int = 0
    candidate_failure_mode: int | None = None

    @classmethod
    def from_input(cls, task_input: dict[str, Any]) -> SimSpec:
        return cls(
            baseline_rate=float(task_input.get("baseline_rate", 0.9)),
            candidate_rate=float(task_input.get("candidate_rate", 0.9)),
            coupling=float(task_input.get("coupling", 0.7)),
            cost_usd=float(task_input.get("cost_usd", 0.02)),
            latency_ms=float(task_input.get("latency_ms", 900.0)),
            steps=int(task_input.get("steps", 4)),
            failure_mode=int(task_input.get("failure_mode", 0)),
            candidate_failure_mode=(
                None
                if task_input.get("candidate_failure_mode") is None
                else int(task_input["candidate_failure_mode"])
            ),
        )


def _latent(seed: int, coupling: float, build: str) -> float:
    """The uniform draw that decides this replicate's outcome.

    Both builds compute the shared draw and the coupling decision from the seed
    alone, so they agree on when they are coupled. When they are not, each takes
    its own stream. Mixing two uniform draws arithmetically would have been
    simpler and wrong: it changes the marginal distribution, so the task's pass
    rate would no longer be the number the caller asked for.
    """
    shared_rng = random.Random(seed ^ _LATENT_SALT)
    shared_draw = shared_rng.random()
    if shared_rng.random() < coupling:
        return shared_draw
    offset = 1 if build == "baseline" else 2
    return random.Random((seed * 2654435761 + offset) & 0xFFFFFFFF).random()


def _trajectory(rng: random.Random, spec: SimSpec, passed: bool, mode: int) -> tuple[Step, ...]:
    """A plausible-looking sequence of tool calls, diverging when a run fails."""
    names = ["plan", "search", "fetch", "compute", "verify", "answer"]
    kept = names[: max(spec.steps, 2)]
    steps: list[Step] = []
    for index, name in enumerate(kept):
        steps.append(
            Step(
                index=index,
                kind="tool_call" if name != "answer" else "message",
                name=name,
                ok=True,
                args={"q": f"{name}-{rng.randrange(1000)}"},
                latency_ms=spec.latency_ms / max(len(kept), 1),
            )
        )
    if passed:
        return tuple(steps)

    kind, detail = _FAILURE_MODES[mode % len(_FAILURE_MODES)]
    cut = rng.randrange(1, max(len(steps), 2))
    steps = steps[:cut]
    if kind == "loop":
        repeat = steps[-1]
        steps.extend(
            Step(
                index=cut + i,
                kind=repeat.kind,
                name=repeat.name,
                ok=True,
                args=repeat.args,
                latency_ms=repeat.latency_ms,
            )
            for i in range(2)
        )
    steps.append(
        Step(
            index=len(steps),
            kind="error",
            name=kind,
            ok=False,
            detail=f"{detail} (case {rng.randrange(100)})",
            latency_ms=spec.latency_ms / 4.0,
        )
    )
    return tuple(steps)


def simulate_run(task_input: dict[str, Any], seed: int, build: str) -> RunOutcome:
    """Run one replicate of the fake agent."""
    spec = SimSpec.from_input(task_input)
    rate = spec.baseline_rate if build == "baseline" else spec.candidate_rate
    draw = _latent(seed, spec.coupling, build)
    passed = draw < rate

    rng = random.Random((seed, build).__hash__() & 0xFFFFFFFF)
    mode = spec.failure_mode
    if build == "candidate" and spec.candidate_failure_mode is not None and not passed:
        # A regression that introduces a *new* way of failing is the case the
        # trajectory diff is supposed to make obvious.
        mode = spec.candidate_failure_mode
    steps = _trajectory(rng, spec, passed, mode)
    error = None if passed else steps[-1].detail
    jitter = 0.75 + 0.5 * rng.random()
    return RunOutcome(
        passed=passed,
        score=float(draw < rate) * (0.6 + 0.4 * rng.random()),
        error=error,
        cost_usd=spec.cost_usd * jitter,
        latency_ms=spec.latency_ms * jitter,
        steps=steps,
        metadata={"build": build, "seed": seed},
    )


def baseline(task_input: dict[str, Any], seed: int, **_: Any) -> RunOutcome:
    """Entry point for the baseline build, referenced from suite YAML."""
    return simulate_run(task_input, seed, "baseline")


def candidate(task_input: dict[str, Any], seed: int, **_: Any) -> RunOutcome:
    """Entry point for the candidate build, referenced from suite YAML."""
    return simulate_run(task_input, seed, "candidate")
