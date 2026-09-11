"""Accepting a gate result from someone's CI.

The payload is exactly what ``arbiter gate --json`` writes, plus a little
context about where the run came from. It arrives from a machine we do not
control, over a token that may have leaked, so everything here is validated
rather than trusted: unknown fields are ignored, strings are length-capped, and
the task list is bounded.

The schema is deliberately tolerant about *missing* fields and strict about
malformed ones. A newer arbiter adding a field should not break ingestion; a
field that is present but nonsense should be rejected loudly.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from .models import GateRun, Project, TaskResult

__all__ = ["MAX_TASKS", "GateRunPayload", "RunContext", "TaskPayload", "store_run"]

MAX_TASKS = 5000
_VERDICTS = {"pass", "warn", "fail", "error"}

Str200 = Annotated[str, Field(max_length=200)]


class TaskPayload(BaseModel):
    """One task's result, as arbiter emits it."""

    model_config = ConfigDict(extra="ignore")

    task_id: Annotated[str, Field(min_length=1, max_length=200)]
    verdict: Annotated[str, Field(max_length=20)] = "inconclusive"
    stop_reason: Annotated[str, Field(max_length=32)] = ""
    flagged: bool = False
    replicates: Annotated[int, Field(ge=0, le=10_000_000)] = 0
    regressions: Annotated[int, Field(ge=0, le=10_000_000)] = 0
    improvements: Annotated[int, Field(ge=0, le=10_000_000)] = 0
    baseline_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    candidate_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    delta: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    e_value: Annotated[float, Field(ge=0.0)] = 1.0
    anytime_p: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    adjusted_p: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    tags: list[Str200] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _cap_tags(cls, v: list[str]) -> list[str]:
        return v[:20]


class GateRunPayload(BaseModel):
    """A whole gate result."""

    model_config = ConfigDict(extra="ignore")

    suite: Str200 = ""
    verdict: Annotated[str, Field(max_length=20)]
    correction: Annotated[str, Field(max_length=20)] = ""
    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    evidence_threshold: Annotated[float, Field(ge=0.0)] = 0.0
    n_tasks: Annotated[int, Field(ge=0)] = 0
    n_flagged: Annotated[int, Field(ge=0)] = 0
    n_cleared: Annotated[int, Field(ge=0)] = 0
    n_no_evidence: Annotated[int, Field(ge=0)] = 0
    n_incomplete: Annotated[int, Field(ge=0)] = 0
    replicates_run: Annotated[int, Field(ge=0)] = 0
    replicates_reused: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0.0)] = 0.0
    wall_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    infra_error_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    notes: list[Annotated[str, Field(max_length=1000)]] = Field(default_factory=list)
    tasks: list[TaskPayload] = Field(default_factory=list)

    @field_validator("verdict")
    @classmethod
    def _known_verdict(cls, v: str) -> str:
        if v not in _VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}, got {v!r}")
        return v

    @field_validator("tasks")
    @classmethod
    def _cap_tasks(cls, v: list[TaskPayload]) -> list[TaskPayload]:
        if len(v) > MAX_TASKS:
            raise ValueError(f"too many tasks: {len(v)} (limit {MAX_TASKS})")
        return v

    @field_validator("notes")
    @classmethod
    def _cap_notes(cls, v: list[str]) -> list[str]:
        return v[:50]


class RunContext(BaseModel):
    """Where the run came from. Every field optional."""

    model_config = ConfigDict(extra="ignore")

    commit_sha: Annotated[str, Field(max_length=64)] = ""
    branch: Str200 = ""
    pr_number: int | None = Field(default=None, ge=0, le=10_000_000)
    ci_url: Annotated[str, Field(max_length=500)] = ""

    @field_validator("ci_url")
    @classmethod
    def _http_only(cls, v: str) -> str:
        """Reject anything that is not a plain http(s) link.

        This value is rendered as an anchor href, so allowing arbitrary schemes
        would put javascript: one template away from being clickable.
        """
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("ci_url must be an http or https URL")
        return v


class IngestRequest(BaseModel):
    """The body of POST /api/v1/runs."""

    model_config = ConfigDict(extra="ignore")

    result: GateRunPayload
    context: RunContext = Field(default_factory=RunContext)


def store_run(
    session: Session,
    project: Project,
    payload: GateRunPayload,
    context: RunContext,
    raw: dict[str, Any] | None = None,
) -> GateRun:
    """Persist a gate result and its per-task rows.

    Counts come from the tasks actually submitted rather than from the summary
    fields, so a payload whose header disagrees with its body is stored
    consistently with what can be seen in the task table.
    """
    tasks = payload.tasks
    flagged = sum(1 for t in tasks if t.flagged)
    cleared = sum(1 for t in tasks if t.verdict == "pass")
    no_evidence = sum(
        1
        for t in tasks
        if t.verdict == "inconclusive"
        and t.stop_reason in {"futility", "max-replicates"}
    )
    incomplete = sum(1 for t in tasks if t.stop_reason == "budget")

    run = GateRun(
        project_id=project.id,
        suite=payload.suite or project.name,
        verdict=payload.verdict,
        commit_sha=context.commit_sha,
        branch=context.branch,
        pr_number=context.pr_number,
        ci_url=context.ci_url,
        n_tasks=len(tasks) or payload.n_tasks,
        n_flagged=flagged,
        n_cleared=cleared,
        n_no_evidence=no_evidence,
        n_incomplete=incomplete,
        replicates_run=payload.replicates_run,
        replicates_reused=payload.replicates_reused,
        cost_usd=payload.cost_usd,
        wall_seconds=payload.wall_seconds,
        infra_error_rate=payload.infra_error_rate,
        correction=payload.correction,
        alpha=payload.alpha,
        evidence_threshold=payload.evidence_threshold,
        notes=payload.notes,
        payload=raw or {},
    )
    session.add(run)
    session.flush()

    session.add_all(
        [
            TaskResult(
                gate_run_id=run.id,
                project_id=project.id,
                task_id=t.task_id,
                verdict=t.verdict,
                stop_reason=t.stop_reason,
                flagged=t.flagged,
                replicates=t.replicates,
                regressions=t.regressions,
                improvements=t.improvements,
                baseline_rate=t.baseline_rate,
                candidate_rate=t.candidate_rate,
                delta=t.delta,
                e_value=t.e_value,
                anytime_p=t.anytime_p,
                adjusted_p=t.adjusted_p,
                tags=",".join(t.tags)[:1000],
                created_at=run.created_at,
            )
            for t in tasks
        ]
    )
    session.flush()
    return run
