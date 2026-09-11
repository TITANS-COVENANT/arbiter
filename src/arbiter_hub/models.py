"""Database schema.

The shape worth explaining is why task results get their own table rather than
living inside the run's JSON payload.

A pull request comment already tells you what happened on one run. The reason to
keep a history at all is the question you cannot answer from a single run: has
this task been unreliable for a month, is it getting worse, and is the thing that
just went red actually new. Answering that needs task rows you can group by
task_id across runs, so that is how they are stored. The full payload is kept
alongside for anything the columns do not cover.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "ApiToken",
    "Base",
    "GateRun",
    "Membership",
    "Org",
    "Project",
    "Role",
    "TaskResult",
    "User",
    "utcnow",
]


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    # Maps the `dict[str, Any]` annotation onto a JSON column, so the payload
    # fields below can be declared with an ordinary Python type.
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


class Role(StrEnum):
    OWNER = "owner"
    MEMBER = "member"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    github_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    login: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    email: Mapped[str] = mapped_column(String(320), default="")
    avatar_url: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def display_name(self) -> str:
        return self.name or self.login


class Org(Base):
    """A billing and isolation boundary. Every user gets one on first sign-in."""

    __tablename__ = "orgs"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    is_personal: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    memberships: Mapped[list[Membership]] = relationship(
        back_populates="org", cascade="all, delete-orphan"
    )
    projects: Mapped[list[Project]] = relationship(
        back_populates="org", cascade="all, delete-orphan"
    )


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "org_id", name="uq_membership"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    role: Mapped[Role] = mapped_column(String(20), default=Role.OWNER)

    user: Mapped[User] = relationship(back_populates="memberships")
    org: Mapped[Org] = relationship(back_populates="memberships")


class Project(Base):
    """One eval suite being tracked, usually one per repository."""

    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("org_id", "slug", name="uq_project_slug"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id", ondelete="CASCADE"), index=True)
    slug: Mapped[str] = mapped_column(String(80), index=True)
    name: Mapped[str] = mapped_column(String(200))
    repo_full_name: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    org: Mapped[Org] = relationship(back_populates="projects")
    tokens: Mapped[list[ApiToken]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    runs: Mapped[list[GateRun]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.repo_full_name}" if self.repo_full_name else ""


class ApiToken(Base):
    """A CI credential.

    Only the hash is stored. The plaintext is shown once at creation and cannot
    be recovered, because a dashboard that can show you your own token can also
    show it to whoever gets into your session.
    """

    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120), default="ci")
    prefix: Mapped[str] = mapped_column(String(16), index=True)
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped[Project] = relationship(back_populates="tokens")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class GateRun(Base):
    """One `arbiter gate` result."""

    __tablename__ = "gate_runs"
    __table_args__ = (
        Index("ix_gate_runs_project_created", "project_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    suite: Mapped[str] = mapped_column(String(200), default="")
    verdict: Mapped[str] = mapped_column(String(20), index=True)

    # Where the run came from. All optional, because a suite can be gated
    # outside a pull request.
    commit_sha: Mapped[str] = mapped_column(String(64), default="", index=True)
    branch: Mapped[str] = mapped_column(String(200), default="")
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ci_url: Mapped[str] = mapped_column(String(500), default="")

    n_tasks: Mapped[int] = mapped_column(Integer, default=0)
    n_flagged: Mapped[int] = mapped_column(Integer, default=0)
    n_cleared: Mapped[int] = mapped_column(Integer, default=0)
    n_no_evidence: Mapped[int] = mapped_column(Integer, default=0)
    n_incomplete: Mapped[int] = mapped_column(Integer, default=0)

    replicates_run: Mapped[int] = mapped_column(Integer, default=0)
    replicates_reused: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    wall_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    infra_error_rate: Mapped[float] = mapped_column(Float, default=0.0)

    correction: Mapped[str] = mapped_column(String(20), default="")
    alpha: Mapped[float] = mapped_column(Float, default=0.0)
    evidence_threshold: Mapped[float] = mapped_column(Float, default=0.0)

    notes: Mapped[dict[str, Any]] = mapped_column(JSON, default=list)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    project: Mapped[Project] = relationship(back_populates="runs")
    tasks: Mapped[list[TaskResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    @property
    def passed(self) -> bool:
        return self.verdict in {"pass", "warn"}

    @property
    def short_sha(self) -> str:
        return self.commit_sha[:7]


class TaskResult(Base):
    """One task's outcome within one run.

    Denormalised out of the payload on purpose, so that "how has this task
    behaved over the last thirty runs" is an index scan rather than a JSON
    crawl. That query is the entire reason this service exists.
    """

    __tablename__ = "task_results"
    __table_args__ = (
        Index("ix_task_results_project_task", "project_id", "task_id"),
        Index("ix_task_results_run", "gate_run_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    gate_run_id: Mapped[int] = mapped_column(
        ForeignKey("gate_runs.id", ondelete="CASCADE"), index=True
    )
    # Duplicated from the run so per-task history does not need a join.
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )

    task_id: Mapped[str] = mapped_column(String(200), index=True)
    verdict: Mapped[str] = mapped_column(String(20))
    stop_reason: Mapped[str] = mapped_column(String(32), default="")
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    replicates: Mapped[int] = mapped_column(Integer, default=0)
    regressions: Mapped[int] = mapped_column(Integer, default=0)
    improvements: Mapped[int] = mapped_column(Integer, default=0)
    baseline_rate: Mapped[float] = mapped_column(Float, default=0.0)
    candidate_rate: Mapped[float] = mapped_column(Float, default=0.0)
    delta: Mapped[float] = mapped_column(Float, default=0.0)
    e_value: Mapped[float] = mapped_column(Float, default=1.0)
    anytime_p: Mapped[float] = mapped_column(Float, default=1.0)
    adjusted_p: Mapped[float] = mapped_column(Float, default=1.0)
    tags: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    run: Mapped[GateRun] = relationship(back_populates="tasks")

    @property
    def tag_list(self) -> list[str]:
        return [t for t in self.tags.split(",") if t]
