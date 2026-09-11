"""The data an eval target hands back, and the shape arbiter stores it in."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RunOutcome", "Step", "Task", "seed_for"]


def _digest(value: Any) -> str:
    """Stable short digest of an arbitrary JSON-able value.

    Tool arguments are digested rather than stored verbatim in the trajectory
    signature so that alignment compares *shape* of behaviour. Two runs that
    both called ``search(query=...)`` with different queries should align as the
    same step; the payloads are kept separately for the humans reading the diff.
    """
    try:
        blob = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(value)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=6).hexdigest()


@dataclass(frozen=True)
class Step:
    """One observable action inside a run.

    Deliberately minimal. arbiter does not care what framework produced the
    trajectory, only that steps have a kind, a name and an outcome, which is
    enough to align two runs and show where they diverged.
    """

    index: int
    kind: str
    name: str
    ok: bool = True
    args: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    latency_ms: float = 0.0

    @property
    def signature(self) -> str:
        """What alignment matches on."""
        return f"{self.kind}:{self.name}:{'ok' if self.ok else 'err'}"

    @property
    def args_digest(self) -> str:
        return _digest(self.args)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "name": self.name,
            "ok": self.ok,
            "args": self.args,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Step:
        return cls(
            index=int(raw.get("index", 0)),
            kind=str(raw.get("kind", "step")),
            name=str(raw.get("name", "")),
            ok=bool(raw.get("ok", True)),
            args=dict(raw.get("args") or {}),
            detail=str(raw.get("detail", "")),
            latency_ms=float(raw.get("latency_ms", 0.0)),
        )


@dataclass(frozen=True)
class RunOutcome:
    """Result of running one task once.

    ``passed`` is what the binary tests consume and ``score`` what the scalar
    ones consume; a target may supply either or both. ``error`` is free text and
    gets normalised into a failure signature by the clustering in
    :mod:`arbiter.diff.cluster`, so it is worth making it descriptive.
    """

    passed: bool
    score: float | None = None
    error: str | None = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    steps: tuple[Step, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(step.signature for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "error": self.error,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunOutcome:
        score = raw.get("score")
        return cls(
            passed=bool(raw.get("passed", False)),
            score=None if score is None else float(score),
            error=raw.get("error"),
            cost_usd=float(raw.get("cost_usd", 0.0)),
            latency_ms=float(raw.get("latency_ms", 0.0)),
            steps=tuple(Step.from_dict(s) for s in raw.get("steps") or []),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Task:
    """One eval case."""

    id: str
    input: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "input": self.input, "tags": list(self.tags), "weight": self.weight}


def seed_for(task_id: str, replicate: int, salt: str = "") -> int:
    """Deterministic seed for a (task, replicate) cell.

    The whole paired design rests on this: the baseline and the candidate must
    receive the *same* seed for the same cell, so that whatever randomness the
    target can control is held fixed across the two builds and only the change
    under test varies. Derived from a hash rather than a counter so that adding
    a task to the suite does not shift every other task's seeds and invalidate
    the cached baseline runs.
    """
    material = f"{salt}\x00{task_id}\x00{replicate}".encode()
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & 0x7FFFFFFF
