"""Explaining a flagged task: what failed, how often, and where it diverged."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..runner.types import RunOutcome
from .align import Alignment, AlignOp, OpKind, align
from .cluster import (
    FailureMode,
    ModeDelta,
    cluster_failures,
    compare_modes,
    normalise_error,
    step_name_counts,
)

__all__ = [
    "AlignOp",
    "Alignment",
    "FailureMode",
    "ModeDelta",
    "OpKind",
    "TaskDiff",
    "align",
    "cluster_failures",
    "compare_modes",
    "diff_task",
    "normalise_error",
    "step_name_counts",
]


@dataclass
class TaskDiff:
    """Everything known about how one task's behaviour changed."""

    task_id: str
    baseline_runs: int
    candidate_runs: int
    baseline_passes: int
    candidate_passes: int
    modes: list[ModeDelta] = field(default_factory=list)
    alignment: Alignment | None = None
    representative_replicate: int | None = None
    representative_seed: int | None = None

    @property
    def baseline_rate(self) -> float:
        return self.baseline_passes / self.baseline_runs if self.baseline_runs else 0.0

    @property
    def candidate_rate(self) -> float:
        return self.candidate_passes / self.candidate_runs if self.candidate_runs else 0.0

    @property
    def new_modes(self) -> list[ModeDelta]:
        return [m for m in self.modes if m.is_new]

    @property
    def fixed_modes(self) -> list[ModeDelta]:
        return [m for m in self.modes if m.is_fixed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "baseline_runs": self.baseline_runs,
            "candidate_runs": self.candidate_runs,
            "baseline_rate": self.baseline_rate,
            "candidate_rate": self.candidate_rate,
            "representative_replicate": self.representative_replicate,
            "modes": [
                {
                    "signature": m.signature,
                    "baseline": m.baseline_count,
                    "candidate": m.candidate_count,
                    "delta": m.delta,
                    "new": m.is_new,
                    "example": m.example,
                }
                for m in self.modes
            ],
            "alignment": self.alignment.summary() if self.alignment else None,
        }


def diff_task(
    task_id: str,
    baseline_runs: Sequence[tuple[int, RunOutcome]],
    candidate_runs: Sequence[tuple[int, RunOutcome]],
) -> TaskDiff:
    """Assemble the explanation for one task.

    The representative pair is chosen deliberately: a replicate where the
    baseline passed and the candidate failed, at the same seed, is the single
    clearest exhibit of what the change broke. Any other pairing is either a
    shared failure, which is not about this change, or a coincidence.
    """
    baseline_by_replicate = dict(baseline_runs)
    candidate_by_replicate = dict(candidate_runs)
    base_outcomes = [outcome for _, outcome in baseline_runs]
    cand_outcomes = [outcome for _, outcome in candidate_runs]

    representative: int | None = None
    for replicate, base in sorted(baseline_by_replicate.items()):
        candidate = candidate_by_replicate.get(replicate)
        if candidate is not None and base.passed and not candidate.passed:
            representative = replicate
            break
    if representative is None:
        shared = sorted(set(baseline_by_replicate) & set(candidate_by_replicate))
        representative = shared[0] if shared else None

    alignment = None
    if representative is not None:
        alignment = align(
            baseline_by_replicate[representative].steps,
            candidate_by_replicate[representative].steps,
        )

    return TaskDiff(
        task_id=task_id,
        baseline_runs=len(base_outcomes),
        candidate_runs=len(cand_outcomes),
        baseline_passes=sum(1 for o in base_outcomes if o.passed),
        candidate_passes=sum(1 for o in cand_outcomes if o.passed),
        modes=compare_modes(base_outcomes, cand_outcomes),
        alignment=alignment,
        representative_replicate=representative,
    )
