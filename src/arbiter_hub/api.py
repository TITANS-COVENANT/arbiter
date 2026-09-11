"""The machine-facing API.

One endpoint matters: ``POST /api/v1/runs``, which takes exactly what
``arbiter gate --json`` writes. Keeping the wire format identical to the CLI's
own output means there is no translation layer to drift, and anyone can post a
result with curl if they would rather not use the ``--publish`` flag.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import get_db
from .deps import require_token_project
from .ingest import IngestRequest, store_run
from .models import GateRun, Project

__all__ = ["router"]

router = APIRouter(prefix="/api/v1", tags=["api"])


class RunAccepted(BaseModel):
    id: int
    verdict: str
    url: str
    n_tasks: int
    n_flagged: int


class ProjectInfo(BaseModel):
    org: str
    project: str
    name: str
    url: str


def _run_url(settings: Settings, project: Project, run: GateRun) -> str:
    org_slug = project.org.slug if project.org else ""
    return f"{settings.base_url.rstrip('/')}/p/{org_slug}/{project.slug}/runs/{run.id}"


@router.post("/runs", response_model=RunAccepted, status_code=status.HTTP_201_CREATED)
def submit_run(
    body: IngestRequest,
    project: Project = Depends(require_token_project),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> RunAccepted:
    """Record a gate result against the token's project."""
    raw: dict[str, Any] = body.result.model_dump(mode="json")
    run = store_run(db, project, body.result, body.context, raw=raw)
    db.commit()
    return RunAccepted(
        id=run.id,
        verdict=run.verdict,
        url=_run_url(settings, project, run),
        n_tasks=run.n_tasks,
        n_flagged=run.n_flagged,
    )


@router.get("/runs/{run_id}")
def get_run(
    run_id: int,
    project: Project = Depends(require_token_project),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Read a run back. Scoped to the token's project."""
    run = db.scalar(
        select(GateRun).where(GateRun.id == run_id, GateRun.project_id == project.id)
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    return {
        "id": run.id,
        "suite": run.suite,
        "verdict": run.verdict,
        "commit_sha": run.commit_sha,
        "branch": run.branch,
        "pr_number": run.pr_number,
        "n_tasks": run.n_tasks,
        "n_flagged": run.n_flagged,
        "replicates_run": run.replicates_run,
        "replicates_reused": run.replicates_reused,
        "cost_usd": run.cost_usd,
        "created_at": run.created_at.isoformat(),
        "tasks": [
            {
                "task_id": t.task_id,
                "verdict": t.verdict,
                "stop_reason": t.stop_reason,
                "flagged": t.flagged,
                "replicates": t.replicates,
                "regressions": t.regressions,
                "improvements": t.improvements,
                "delta": t.delta,
                "e_value": t.e_value,
                "adjusted_p": t.adjusted_p,
            }
            for t in run.tasks
        ],
    }


@router.get("/projects/me", response_model=ProjectInfo)
def whoami(
    project: Project = Depends(require_token_project),
    settings: Settings = Depends(get_settings),
) -> ProjectInfo:
    """What this token can write to.

    Exists so that a CI job failing to publish can be diagnosed with one curl
    rather than by guessing whether the token or the URL is wrong.
    """
    org_slug = project.org.slug if project.org else ""
    return ProjectInfo(
        org=org_slug,
        project=project.slug,
        name=project.name,
        url=f"{settings.base_url.rstrip('/')}/p/{org_slug}/{project.slug}",
    )


health_router = APIRouter(tags=["meta"])


@health_router.get("/health")
def health(request: Request, db: Session = Depends(get_db)) -> dict[str, str]:
    """Liveness plus a real database round trip."""
    db.execute(select(1))
    return {"status": "ok"}
