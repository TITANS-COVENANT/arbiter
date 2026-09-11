"""Suite configuration: the YAML file that describes what to run and how to decide.

One file holds the tasks, the two builds being compared, the statistical policy
and the budget. It is hashed to produce a variant identity, which is what makes
baseline reuse safe: change the model, the prompt or the target command and the
hash moves, so yesterday's cached baseline runs stop being used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .errors import ConfigError
from .runner.types import Task

__all__ = [
    "BudgetConfig",
    "GateConfig",
    "StatsConfig",
    "SuiteConfig",
    "TargetConfig",
    "TaskConfig",
    "load_suite",
]


class TaskConfig(BaseModel):
    """One eval case as it appears in YAML."""

    id: str
    input: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0

    def to_task(self) -> Task:
        return Task(id=self.id, input=self.input, tags=tuple(self.tags), weight=self.weight)


class TargetConfig(BaseModel):
    """How to invoke one build of the thing under test.

    ``python`` imports ``module:callable`` in-process, which is the fastest path
    and the one the tests use. ``subprocess`` runs a command and reads a JSON
    object from stdout. ``http`` posts the task to an endpoint. All three
    receive the task input and a seed, and are expected to return the fields of
    :class:`~arbiter.runner.types.RunOutcome`.
    """

    kind: Literal["python", "subprocess", "http"] = "python"
    ref: str | None = None
    command: list[str] | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 120.0
    max_retries: int = 2
    concurrency: int = 8
    rate_limit_per_s: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_kind_fields(self) -> TargetConfig:
        required = {"python": "ref", "subprocess": "command", "http": "url"}[self.kind]
        if getattr(self, required) is None:
            raise ValueError(f"target of kind '{self.kind}' requires '{required}'")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        return self


class StatsConfig(BaseModel):
    """Statistical policy.

    ``mode`` picks what evidence the gate reads. ``binary`` runs the paired
    McNemar sequential test on pass/fail. ``score`` runs an always-valid
    confidence sequence on paired score differences.

    ``mde`` and ``odds_ratio`` both describe "how big a regression is worth
    catching", from different angles. ``odds_ratio`` is what the binary test
    actually uses; ``mde`` drives planning and the fixed-sample comparison.
    """

    mode: Literal["binary", "score"] = "binary"
    alpha: float = 0.05
    beta: float = 0.10
    mde: float = 0.15
    odds_ratio: float = 3.0
    score_tolerance: float = 0.05
    score_range: float = 1.0
    min_replicates: int = 4
    max_replicates: int = 200
    correction: Literal["e-bh", "bh", "holm", "none"] = "e-bh"
    futility_confidence: float = 0.01
    """Confidence level for the futility bound that abandons hopeless tasks.

    Lower means more cautious: the bound on how often the two builds will
    disagree in future gets wider, so fewer tasks are written off early. It
    trades budget for power and cannot cause a false flag, only a miss."""

    @field_validator("alpha", "beta")
    @classmethod
    def _in_unit(cls, v: float) -> float:
        if not 0.0 < v < 0.5:
            raise ValueError(f"must lie in (0, 0.5), got {v}")
        return v

    @model_validator(mode="after")
    def _check(self) -> StatsConfig:
        if not 0.0 < self.mde < 1.0:
            raise ValueError(f"mde must lie in (0, 1), got {self.mde}")
        if self.min_replicates > self.max_replicates:
            raise ValueError("min_replicates cannot exceed max_replicates")
        return self


class BudgetConfig(BaseModel):
    """Hard ceilings on what a gate run may spend.

    Whichever binds first stops the run. Tasks still undecided at that point are
    reported inconclusive rather than silently passed, so an underfunded gate
    looks underfunded instead of looking green.
    """

    max_replicates: int | None = None
    max_cost_usd: float | None = None
    max_seconds: float | None = None
    batch_size: int = 16

    @field_validator("batch_size")
    @classmethod
    def _positive_batch(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        return v


class GateConfig(BaseModel):
    """What the exit code means."""

    on_inconclusive: Literal["pass", "fail", "warn"] = "warn"
    fail_on_any_flag: bool = True
    max_flagged_tasks: int = 0
    allocator: Literal["round-robin", "cheapest-to-close", "successive-halving"] = "round-robin"
    """Even spend is the default because it measured best; see allocator.py."""
    max_infra_error_rate: float = 0.25
    """Above this share of unrunnable replicates the gate reports an error.

    A harness that cannot reach the target is not evidence about the candidate,
    and a gate that goes green because most of its runs never happened is worse
    than no gate at all."""


class SuiteConfig(BaseModel):
    """A complete gate definition."""

    name: str = "suite"
    seed_salt: str = ""
    store: str = ".arbiter/runs.sqlite"
    tasks: list[TaskConfig]
    baseline: TargetConfig
    candidate: TargetConfig
    stats: StatsConfig = Field(default_factory=StatsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    gate: GateConfig = Field(default_factory=GateConfig)

    @field_validator("tasks")
    @classmethod
    def _non_empty_unique(cls, v: list[TaskConfig]) -> list[TaskConfig]:
        if not v:
            raise ValueError("a suite needs at least one task")
        ids = [t.id for t in v]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate task ids: {sorted(duplicates)}")
        return v

    def to_tasks(self) -> list[Task]:
        return [t.to_task() for t in self.tasks]

    def variant_id(self, which: Literal["baseline", "candidate"]) -> str:
        """Identity of one build, for cache lookup.

        Covers the target definition and the seed salt, because those decide
        what a stored run *means*. It deliberately excludes the statistical
        policy and the budget: tightening alpha should reuse existing runs, not
        throw them away.
        """
        target = self.baseline if which == "baseline" else self.candidate
        material = json.dumps(
            {
                "suite": self.name,
                "salt": self.seed_salt,
                "target": target.model_dump(mode="json"),
                "tasks": [t.model_dump(mode="json") for t in self.tasks],
            },
            sort_keys=True,
        )
        return hashlib.blake2b(material.encode("utf-8"), digest_size=10).hexdigest()


def load_suite(path: str | Path) -> SuiteConfig:
    """Read and validate a suite YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"suite file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"suite file must contain a mapping at the top level: {p}")
    return SuiteConfig.model_validate(raw)
