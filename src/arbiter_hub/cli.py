"""Command line for the hub.

    arbiter-hub serve     run it
    arbiter-hub init-db   create the tables
    arbiter-hub demo      seed a project with synthetic history, to look around
"""

from __future__ import annotations

import datetime as dt
import os
import random
from typing import Annotated

import typer
from rich.console import Console

from . import __version__
from .config import get_settings
from .db import build_engine, create_all, session_scope
from .ingest import GateRunPayload, RunContext, TaskPayload, store_run
from .models import ApiToken, Membership, Org, Project, Role, User, utcnow
from .security import generate_token, hash_token, slugify, token_prefix

app = typer.Typer(
    name="arbiter-hub",
    help="The service your CI reports arbiter verdicts into.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    console.print(f"arbiter-hub {__version__}")


@app.command("init-db")
def init_db() -> None:
    """Create any missing tables."""
    settings = get_settings()
    create_all(build_engine(settings))
    console.print(f"schema ready at [bold]{settings.database_url}[/bold]")


def _override(**values: str | None) -> None:
    """Push CLI options into the environment before settings are first read.

    Settings are read from the environment exactly once and cached, so an option
    that should behave like configuration has to arrive that way. Doing it here
    keeps a single source of truth instead of two parallel config paths.
    """
    for key, value in values.items():
        if value is not None:
            os.environ[f"ARBITER_HUB_{key.upper()}"] = value
    get_settings.cache_clear()


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on code changes.")] = False,
    dev: Annotated[
        bool,
        typer.Option(
            "--dev",
            help="Local mode: enables /auth/dev, which signs anyone in as anyone.",
        ),
    ] = False,
    database_url: Annotated[
        str | None, typer.Option("--database-url", help="Override the database.")
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Public URL, used in links and OAuth.")
    ] = None,
) -> None:
    """Run the web application."""
    import uvicorn

    _override(
        dev_auth="1" if dev else None,
        database_url=database_url,
        base_url=base_url or (f"http://{host}:{port}" if dev else None),
    )
    settings = get_settings()
    create_all(build_engine(settings))

    if not settings.github_configured and not settings.dev_auth:
        console.print(
            "[yellow]No GitHub OAuth credentials and dev auth is off, so nobody can "
            "sign in.[/yellow]\nEither register an OAuth app and set "
            "ARBITER_HUB_GITHUB_CLIENT_ID and ARBITER_HUB_GITHUB_CLIENT_SECRET, or "
            "set ARBITER_HUB_DEV_AUTH=1 for local use."
        )
    if settings.dev_auth:
        console.print(
            "[yellow]Dev auth is on: /auth/dev signs anyone in as anyone. "
            "Never expose this to a network you do not control.[/yellow]"
        )

    uvicorn.run(
        "arbiter_hub.app:build",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def demo(
    login: Annotated[str, typer.Option("--login", help="Account handle to seed.")] = "dev",
    project_name: Annotated[str, typer.Option("--project")] = "Support agent evals",
    runs: Annotated[int, typer.Option("--runs", help="How many runs of history.")] = 24,
    seed: Annotated[int, typer.Option("--seed")] = 7,
) -> None:
    """Seed an account with synthetic history so the dashboard has something in it.

    The generated history is deliberately not uniform: one task is chronically
    flagged, one is genuinely flaky, and the rest are stable. Those three cases
    are what the views exist to tell apart, so a demo where everything looks the
    same would not show whether they work.
    """
    rng = random.Random(seed)
    settings = get_settings()
    create_all(build_engine(settings))

    stable = [f"task_{i:02d}" for i in range(8)]
    chronic = "refund_multi_hop"
    flaky = "tool_retry_timeout"

    with session_scope() as session:
        handle = slugify(login, fallback="dev")
        user = session.query(User).filter(User.login == handle).one_or_none()
        if user is None:
            lowest = session.query(User.github_id).order_by(User.github_id).limit(1).scalar() or 0
            user = User(
                github_id=min(lowest, 0) - 1,
                login=handle,
                name=handle.replace("-", " ").title(),
                email=f"{handle}@example.invalid",
            )
            session.add(user)
            session.flush()
        org = session.query(Org).join(Membership).filter(Membership.user_id == user.id).first()
        if org is None:
            org = Org(slug=slugify(handle, fallback="org"), name=user.display_name)
            session.add(org)
            session.flush()
            session.add(Membership(user_id=user.id, org_id=org.id, role=Role.OWNER))
            session.flush()

        slug = slugify(project_name)
        project = (
            session.query(Project)
            .filter(Project.org_id == org.id, Project.slug == slug)
            .one_or_none()
        )
        if project is None:
            project = Project(
                org_id=org.id, slug=slug, name=project_name, repo_full_name="acme/support-agent"
            )
            session.add(project)
            session.flush()

        token = generate_token()
        session.add(
            ApiToken(
                project_id=project.id,
                name="demo",
                prefix=token_prefix(token),
                token_hash=hash_token(token),
            )
        )

        started = utcnow() - dt.timedelta(days=runs)
        for index in range(runs):
            tasks: list[TaskPayload] = []
            for task_id in stable:
                tasks.append(_clean_task(task_id, rng))
            # Chronic: broken from run 6 onward and never fixed.
            tasks.append(
                _broken_task(chronic, rng) if index >= 6 else _clean_task(chronic, rng)
            )
            # Flaky: flips roughly every other run, with no underlying change.
            tasks.append(
                _broken_task(flaky, rng) if rng.random() < 0.45 else _clean_task(flaky, rng)
            )

            flagged = sum(1 for t in tasks if t.flagged)
            payload = GateRunPayload(
                suite=project_name,
                verdict="fail" if flagged else "pass",
                correction="e-bh",
                alpha=0.05,
                evidence_threshold=len(tasks) / 0.05,
                replicates_run=rng.randint(900, 1600),
                replicates_reused=rng.randint(400, 900),
                cost_usd=round(rng.uniform(3.5, 9.0), 2),
                wall_seconds=round(rng.uniform(120, 420), 1),
                tasks=tasks,
            )
            context = RunContext(
                commit_sha=f"{rng.getrandbits(80):020x}",
                branch="main" if index % 4 else f"feature/change-{index}",
                pr_number=None if index % 4 == 0 else 100 + index,
            )
            run = store_run(session, project, payload, context, raw=payload.model_dump(mode="json"))
            # Spread the history backwards so the charts have a time axis.
            run.created_at = started + dt.timedelta(days=index, hours=rng.randint(0, 9))
            for task_row in run.tasks:
                task_row.created_at = run.created_at
            session.flush()

        url = f"{settings.base_url.rstrip('/')}/p/{org.slug}/{project.slug}"

    console.print(f"seeded [bold]{runs}[/bold] runs into [bold]{url}[/bold]")
    console.print(
        f"sign in at [bold]{settings.base_url}/auth/dev?login={handle}[/bold] "
        f"(requires dev auth)"
    )
    # Deliberately a plain print on its own line, in an assignable form. Rich
    # wraps at the terminal width, and a token split across two lines is one a
    # script cannot read back.
    print(f"ARBITER_HUB_TOKEN={token}")


def _clean_task(task_id: str, rng: random.Random) -> TaskPayload:
    return TaskPayload(
        task_id=task_id,
        verdict=rng.choice(["pass", "inconclusive"]),
        stop_reason=rng.choice(["boundary", "futility"]),
        flagged=False,
        replicates=rng.randint(30, 95),
        regressions=rng.randint(0, 3),
        improvements=rng.randint(0, 3),
        baseline_rate=round(rng.uniform(0.86, 0.97), 3),
        candidate_rate=round(rng.uniform(0.86, 0.97), 3),
        delta=round(rng.uniform(-0.03, 0.03), 3),
        e_value=round(rng.uniform(0.05, 2.5), 3),
        anytime_p=round(rng.uniform(0.3, 1.0), 3),
        adjusted_p=round(rng.uniform(0.3, 1.0), 3),
    )


def _broken_task(task_id: str, rng: random.Random) -> TaskPayload:
    baseline = round(rng.uniform(0.88, 0.96), 3)
    candidate = round(baseline - rng.uniform(0.25, 0.5), 3)
    return TaskPayload(
        task_id=task_id,
        verdict="regression",
        stop_reason="boundary",
        flagged=True,
        replicates=rng.randint(14, 30),
        regressions=rng.randint(12, 20),
        improvements=rng.randint(0, 2),
        baseline_rate=baseline,
        candidate_rate=max(candidate, 0.0),
        delta=round(max(candidate, 0.0) - baseline, 3),
        e_value=round(rng.uniform(180, 900), 1),
        anytime_p=round(rng.uniform(0.0005, 0.01), 5),
        adjusted_p=round(rng.uniform(0.001, 0.02), 5),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
