"""Fixtures for the hub tests.

Each test gets its own SQLite file and its own settings, because the app reads
configuration once at startup and caches it. Sharing either between tests turns
an ordering bug into an intermittent failure, which is the worst kind.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from arbiter_hub import db as db_module
from arbiter_hub.app import create_app
from arbiter_hub.config import Settings, get_settings
from arbiter_hub.deps import RateLimiter
from arbiter_hub.models import ApiToken, Membership, Org, Project, Role, User
from arbiter_hub.security import generate_token, hash_token, token_prefix

_ENV_KEYS = [k for k in os.environ if k.startswith("ARBITER_HUB_")]


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A stray ARBITER_HUB_* variable in the developer's shell must not leak in."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    db_module.reset_state()
    yield
    get_settings.cache_clear()
    db_module.reset_state()


@pytest.fixture
def settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ARBITER_HUB_DATABASE_URL", f"sqlite:///{tmp_path / 'hub.sqlite'}")
    monkeypatch.setenv("ARBITER_HUB_DEV_AUTH", "1")
    monkeypatch.setenv("ARBITER_HUB_SECRET_KEY", "test-secret-key-not-for-real-use")
    monkeypatch.setenv("ARBITER_HUB_BASE_URL", "http://testserver")
    get_settings.cache_clear()
    db_module.reset_state()
    return get_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    import arbiter_hub.deps as deps_module

    deps_module._ingest_limiter = RateLimiter(settings.ingest_rate_per_minute)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    db_module.create_all(db_module.build_engine(settings))
    with db_module.session_scope() as session:
        yield session


def make_user(session: Session, login: str) -> User:
    user = User(github_id=abs(hash(login)) % 10_000_000, login=login, name=login.title())
    session.add(user)
    session.flush()
    org = Org(slug=f"{login}-org", name=f"{login} org")
    session.add(org)
    session.flush()
    session.add(Membership(user_id=user.id, org_id=org.id, role=Role.OWNER))
    session.flush()
    return user


def make_project(session: Session, user: User, slug: str = "evals") -> Project:
    org = session.query(Org).join(Membership).filter(Membership.user_id == user.id).one()
    project = Project(org_id=org.id, slug=slug, name=slug.title())
    session.add(project)
    session.flush()
    return project


def make_token(session: Session, project: Project) -> str:
    raw = generate_token()
    session.add(
        ApiToken(
            project_id=project.id,
            name="test",
            prefix=token_prefix(raw),
            token_hash=hash_token(raw),
        )
    )
    session.flush()
    return raw


def sample_result(verdict: str = "fail", flagged_task: str = "broken") -> dict:
    """A payload shaped like what `arbiter gate --json` writes."""
    return {
        "suite": "demo",
        "verdict": verdict,
        "correction": "e-bh",
        "alpha": 0.05,
        "evidence_threshold": 200.0,
        "replicates_run": 480,
        "replicates_reused": 120,
        "cost_usd": 4.25,
        "wall_seconds": 31.0,
        "notes": [],
        "tasks": [
            {
                "task_id": flagged_task,
                "verdict": "regression",
                "stop_reason": "boundary",
                "flagged": verdict == "fail",
                "replicates": 22,
                "regressions": 15,
                "improvements": 1,
                "baseline_rate": 0.94,
                "candidate_rate": 0.31,
                "delta": -0.63,
                "e_value": 410.0,
                "anytime_p": 0.002,
                "adjusted_p": 0.004,
                "tags": ["multi-hop"],
            },
            {
                "task_id": "fine",
                "verdict": "pass",
                "stop_reason": "boundary",
                "flagged": False,
                "replicates": 64,
                "regressions": 1,
                "improvements": 2,
                "baseline_rate": 0.92,
                "candidate_rate": 0.93,
                "delta": 0.01,
                "e_value": 0.2,
                "anytime_p": 0.8,
                "adjusted_p": 0.9,
                "tags": [],
            },
        ],
    }
