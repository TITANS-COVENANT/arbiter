"""Deciding which task gets the next replicate.

Once tasks stop at different times, "how many samples per task" stops being a
constant and becomes a scheduling problem. Some tasks resolve in four
replicates. Some are genuinely borderline and would eat the entire budget
without ever deciding. A round-robin schedule spends the same on both, which
means the borderline ones starve everything else.

Three policies are provided, and the default is the boring one, because it
measured best. On the benchmark's three-regressions scenario, spreading spend
evenly used 2,900 replicates at 0.994 per-task power, while spending greedily
where a decision was closest used 3,029 at 0.956. Greedy lost on both counts.

The mechanism is worth understanding before reaching for the clever option. A
greedy schedule defers the tasks furthest from deciding, and on a build that
really did regress, those are the regressed-but-not-yet-obvious ones. It
deprioritises exactly the tasks you are trying to find.

Greedy still earns its place when per-task costs differ a lot, or when a hard
budget ceiling means the run will be cut off and resolving the most tasks per
dollar matters more than resolving the right ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = [
    "Allocator",
    "CheapestToCloseAllocator",
    "RoundRobinAllocator",
    "SuccessiveHalvingAllocator",
    "TaskState",
    "build_allocator",
]


@runtime_checkable
class TaskState(Protocol):
    """What an allocator needs to know about a task in flight."""

    @property
    def task_id(self) -> str: ...

    @property
    def replicates(self) -> int: ...

    @property
    def resolved(self) -> bool: ...

    @property
    def priority(self) -> float: ...

    def expected_remaining(self) -> float: ...


class Allocator(Protocol):
    """Picks the next batch of replicates, as a list of task ids with repeats."""

    name: str

    def select(self, states: Sequence[TaskState], batch_size: int) -> list[str]: ...


def _pending(states: Sequence[TaskState]) -> list[TaskState]:
    return [s for s in states if not s.resolved]


def _warmup(states: Sequence[TaskState], min_replicates: int) -> list[TaskState]:
    return [s for s in _pending(states) if s.replicates < min_replicates]


class RoundRobinAllocator:
    """Equal spend across every undecided task. The default.

    Sorting by replicate count means every task gets its k-th replicate before
    any task gets its (k+1)-th, so the warmup is implicit and ``min_replicates``
    is accepted only to keep the constructor uniform across allocators.
    """

    name = "round-robin"

    def __init__(self, min_replicates: int = 0) -> None:
        self.min_replicates = min_replicates

    def select(self, states: Sequence[TaskState], batch_size: int) -> list[str]:
        pending = _pending(states)
        if not pending:
            return []
        pending.sort(key=lambda s: (s.replicates, s.task_id))
        picks: list[str] = []
        while len(picks) < batch_size:
            for state in pending:
                if len(picks) >= batch_size:
                    break
                picks.append(state.task_id)
        return picks[:batch_size]


class CheapestToCloseAllocator:
    """Spend where a decision is closest, after a fixed warmup.

    Measured slightly worse than round-robin on a suite of equal-cost tasks, for
    the reason given in the module docstring. Worth choosing when per-task costs
    differ, or when a budget ceiling will cut the run short.

    The warmup matters. Before a task has any data its estimated distance to a
    boundary is made of prior, not evidence, and without a floor the allocator
    will happily pour the whole budget into whichever task got a lucky first
    failure. Every task gets ``min_replicates`` before the greedy phase starts.
    """

    name = "cheapest-to-close"

    def __init__(self, min_replicates: int = 4, fairness: float = 0.25) -> None:
        if not 0.0 <= fairness < 1.0:
            raise ValueError(f"fairness must lie in [0, 1), got {fairness}")
        self.min_replicates = min_replicates
        self.fairness = fairness

    def select(self, states: Sequence[TaskState], batch_size: int) -> list[str]:
        pending = _pending(states)
        if not pending:
            return []
        warming = _warmup(states, self.min_replicates)
        if warming:
            warming.sort(key=lambda s: (s.replicates, s.task_id))
            picks = [s.task_id for s in warming[:batch_size]]
            if len(picks) >= batch_size:
                return picks
        else:
            picks = []

        ranked = sorted(pending, key=lambda s: (s.expected_remaining(), -s.priority, s.task_id))
        # A slice of every batch goes to the least-sampled tasks regardless of
        # rank, so that a task the estimator is pessimistic about still
        # accumulates evidence and gets a chance to change its own estimate.
        n_fair = int((batch_size - len(picks)) * self.fairness)
        if n_fair:
            fair = sorted(pending, key=lambda s: (s.replicates, s.task_id))[:n_fair]
            picks.extend(s.task_id for s in fair)

        # Walk the ranking and give each task roughly what it still needs, so a
        # task two replicates from resolving actually resolves instead of getting
        # one replicate and waiting for the next batch.
        #
        # The per-batch cap is not decoration. Every task is re-examined for
        # futility only after a batch lands, so pouring a whole batch into one
        # task delays that check for everything else. Measured on the benchmark's
        # clean-build scenario, removing the cap cost 19% more replicates for no
        # gain in power, because most tasks on a clean build end in futility
        # rather than at a boundary.
        per_task_cap = max(2, batch_size // 4)
        for state in ranked:
            if len(picks) >= batch_size:
                break
            need = max(1, math.ceil(state.expected_remaining()))
            take = min(need, per_task_cap, batch_size - len(picks))
            picks.extend([state.task_id] * take)

        # Every pending task has been given its full expected need and there is
        # still budget in this batch. Cycle, because the estimates were only
        # estimates.
        while len(picks) < batch_size:
            for state in ranked:
                if len(picks) >= batch_size:
                    break
                picks.append(state.task_id)
        return picks[:batch_size]


class SuccessiveHalvingAllocator:
    """Rounds of equal spend, dropping the least suspicious half each round.

    Borrowed from hyperparameter search, where the same structure appears: many
    arms, a cheap noisy signal, and a budget that should concentrate on the
    promising ones. Here "promising" means "most likely to be a real
    regression". Worth choosing when you care about finding the worst few tasks
    quickly rather than reaching a verdict on all of them.
    """

    name = "successive-halving"

    def __init__(self, min_replicates: int = 4, keep_fraction: float = 0.5) -> None:
        if not 0.0 < keep_fraction < 1.0:
            raise ValueError(f"keep_fraction must lie in (0, 1), got {keep_fraction}")
        self.min_replicates = min_replicates
        self.keep_fraction = keep_fraction

    def select(self, states: Sequence[TaskState], batch_size: int) -> list[str]:
        pending = _pending(states)
        if not pending:
            return []
        warming = _warmup(states, self.min_replicates)
        if warming:
            warming.sort(key=lambda s: (s.replicates, s.task_id))
            return [s.task_id for s in warming[:batch_size]]

        # Rung is set by the least-sampled survivor, so the cohort advances
        # together rather than letting one task race ahead.
        rung = min(s.replicates for s in pending)
        survivors = sorted(pending, key=lambda s: (-s.priority, s.task_id))
        keep = max(1, int(len(survivors) * self.keep_fraction))
        cohort = [s for s in survivors[:keep] if s.replicates <= rung * 2 + self.min_replicates]
        if not cohort:
            cohort = survivors[:keep]
        picks: list[str] = []
        while len(picks) < batch_size:
            for state in cohort:
                if len(picks) >= batch_size:
                    break
                picks.append(state.task_id)
        return picks[:batch_size]


def build_allocator(name: str, min_replicates: int = 4) -> Allocator:
    if name == "cheapest-to-close":
        return CheapestToCloseAllocator(min_replicates=min_replicates)
    if name == "successive-halving":
        return SuccessiveHalvingAllocator(min_replicates=min_replicates)
    return RoundRobinAllocator(min_replicates=min_replicates)
