"""The HTML side.

Server-rendered Jinja, with HTMX for the two or three places where a full page
reload would be silly. No client build step, no API duplication, and the
templates read the same objects the rest of the package uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from .analytics import (
    flag_streak,
    project_summary,
    recent_runs,
    task_history,
    task_stats,
)
from .config import Settings, get_settings
from .db import get_db
from .deps import csrf_token, current_user, require_csrf, require_project, require_user, user_orgs
from .models import ApiToken, GateRun, Project, User, utcnow
from .security import generate_token, hash_token, slugify, token_prefix

__all__ = ["router", "templates"]

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
router = APIRouter()


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _delta(value: float) -> str:
    return f"{value * 100:+.1f} pts" if value else "0"


templates.env.filters["pct"] = _pct
templates.env.filters["delta"] = _delta


def render(
    request: Request,
    name: str,
    context: dict[str, Any] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    """Render a template with the bits every page needs."""
    payload: dict[str, Any] = {
        "request": request,
        "csrf": csrf_token(request),
        "settings": get_settings(),
    }
    payload.update(context or {})
    return templates.TemplateResponse(request, name, payload, status_code=status_code)


def _project_url(project: Project) -> str:
    return f"/p/{project.org.slug}/{project.slug}"


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    user: User | None = Depends(current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Marketing page when signed out, your projects when signed in."""
    if user is None:
        return render(request, "landing.html", {"user": None})

    orgs = user_orgs(db, user)
    org_ids = [o.id for o in orgs]
    projects = (
        list(
            db.scalars(
                select(Project).where(Project.org_id.in_(org_ids)).order_by(Project.name)
            )
        )
        if org_ids
        else []
    )
    cards = []
    for project in projects:
        summary = project_summary(db, project)
        cards.append({"project": project, "summary": summary, "url": _project_url(project)})
    return render(
        request,
        "dashboard.html",
        {"user": user, "orgs": orgs, "cards": cards},
    )


@router.get("/projects/new", response_class=HTMLResponse)
def new_project_form(
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    return render(request, "new_project.html", {"user": user, "orgs": user_orgs(db, user)})


@router.post("/projects/new")
def create_project(
    request: Request,
    name: str = Form(...),
    org_id: int = Form(...),
    repo_full_name: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
    _: None = Depends(require_csrf),
) -> Response:
    """Create a project inside an org the user belongs to."""
    orgs = {o.id: o for o in user_orgs(db, user)}
    org = orgs.get(org_id)
    if org is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "organisation not found")

    clean_name = name.strip()[:200]
    if not clean_name:
        return render(
            request,
            "new_project.html",
            {"user": user, "orgs": list(orgs.values()), "error": "Give the project a name."},
            status_code=400,
        )

    repo = repo_full_name.strip()[:200]
    if repo and repo.count("/") != 1:
        return render(
            request,
            "new_project.html",
            {
                "user": user,
                "orgs": list(orgs.values()),
                "error": "The repository should look like owner/name.",
            },
            status_code=400,
        )

    slug = slugify(clean_name)
    while db.scalar(select(Project).where(Project.org_id == org.id, Project.slug == slug)):
        slug = slugify(f"{clean_name} {utcnow().microsecond}")

    project = Project(org_id=org.id, slug=slug, name=clean_name, repo_full_name=repo)
    db.add(project)
    db.commit()
    return RedirectResponse(f"/p/{org.slug}/{slug}/settings?created=1", status_code=303)


@router.get("/p/{org_slug}/{project_slug}", response_class=HTMLResponse)
def project_overview(
    request: Request,
    project: Project = Depends(require_project),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    summary = project_summary(db, project)
    runs = recent_runs(db, project, limit=20)
    stats = task_stats(db, project)
    return render(
        request,
        "project.html",
        {
            "user": user,
            "project": project,
            "summary": summary,
            "runs": runs,
            "stats": stats,
            "flaky": [s for s in stats if s.is_flaky],
            "chronic": [s for s in stats if s.is_chronic],
            "url": _project_url(project),
        },
    )


@router.get("/p/{org_slug}/{project_slug}/runs/{run_id}", response_class=HTMLResponse)
def run_detail(
    request: Request,
    run_id: int,
    project: Project = Depends(require_project),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    run = db.scalar(
        select(GateRun).where(GateRun.id == run_id, GateRun.project_id == project.id)
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
    tasks = sorted(run.tasks, key=lambda t: (not t.flagged, t.adjusted_p, t.task_id))
    return render(
        request,
        "run.html",
        {
            "user": user,
            "project": project,
            "run": run,
            "tasks": tasks,
            "flagged": [t for t in tasks if t.flagged],
            "url": _project_url(project),
        },
    )


@router.get("/p/{org_slug}/{project_slug}/tasks", response_class=HTMLResponse)
def task_detail(
    request: Request,
    id: str = "",
    project: Project = Depends(require_project),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """History for one task. The view that justifies storing any of this."""
    if not id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "which task?")
    history = task_history(db, project, id)
    if not history:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no recorded runs for that task")
    stats = {s.task_id: s for s in task_stats(db, project)}
    return render(
        request,
        "task.html",
        {
            "user": user,
            "project": project,
            "task_id": id,
            "history": history,
            "stat": stats.get(id),
            "streak": flag_streak(history),
            "url": _project_url(project),
        },
    )


@router.get("/p/{org_slug}/{project_slug}/settings", response_class=HTMLResponse)
def project_settings(
    request: Request,
    created: int = 0,
    project: Project = Depends(require_project),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    tokens = sorted(project.tokens, key=lambda t: t.created_at, reverse=True)
    # A freshly minted token is handed over exactly once, through the session,
    # and removed on read so a refresh does not show it again.
    fresh = request.session.pop("fresh_token", None)
    return render(
        request,
        "settings.html",
        {
            "user": user,
            "project": project,
            "tokens": tokens,
            "fresh_token": fresh,
            "just_created": bool(created),
            "ingest_url": f"{settings.base_url.rstrip('/')}/api/v1/runs",
            "url": _project_url(project),
        },
    )


@router.post("/p/{org_slug}/{project_slug}/tokens")
def create_api_token(
    request: Request,
    name: str = Form("ci"),
    project: Project = Depends(require_project),
    db: Session = Depends(get_db),
    _: None = Depends(require_csrf),
) -> Response:
    token = generate_token()
    db.add(
        ApiToken(
            project_id=project.id,
            name=name.strip()[:120] or "ci",
            prefix=token_prefix(token),
            token_hash=hash_token(token),
        )
    )
    db.commit()
    request.session["fresh_token"] = token
    return RedirectResponse(f"{_project_url(project)}/settings", status_code=303)


@router.post("/p/{org_slug}/{project_slug}/tokens/{token_id}/revoke")
def revoke_api_token(
    token_id: int,
    project: Project = Depends(require_project),
    db: Session = Depends(get_db),
    _: None = Depends(require_csrf),
) -> Response:
    token = db.scalar(
        select(ApiToken).where(ApiToken.id == token_id, ApiToken.project_id == project.id)
    )
    if token is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "token not found")
    token.revoked_at = utcnow()
    db.commit()
    return RedirectResponse(f"{_project_url(project)}/settings", status_code=303)


@router.post("/p/{org_slug}/{project_slug}/delete")
def delete_project(
    confirm: str = Form(""),
    project: Project = Depends(require_project),
    db: Session = Depends(get_db),
    _: None = Depends(require_csrf),
) -> Response:
    """Delete a project and everything under it.

    Requires the project's slug typed back, because the cascade takes every run
    and every task row with it and there is no undo.
    """
    if confirm.strip() != project.slug:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"type the project slug ({project.slug}) to confirm deletion",
        )
    db.delete(project)
    db.commit()
    return RedirectResponse("/", status_code=303)
