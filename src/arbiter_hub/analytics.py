"""The queries that make a history worth keeping.

A pull request comment already answers "did this run pass". The questions that
need a history are different, and there are three worth answering:

**Which tasks keep flipping?** A task that alternates between flagged and clear
across runs is telling you its signal is unstable, not that your build keeps
breaking and healing. That is a flaky eval and it should be fixed or retired.

**Which tasks are chronically flagged?** Flagged in most runs, rarely flipping.
That is not flakiness, it is a regression nobody got round to fixing, and it
should stop being reported as news.

**Which tasks cost the most?** Replicates per decision, averaged. These are the
tasks where a better seed or a larger declared effect would buy the most.

Keeping the first two apart matters. Lumping them into one "unreliable tasks"
list is how a real unfixed regression ends up filed under flakiness and ignored.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from itertools import pairwise

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import GateRun, Project, TaskResult

__all__ = [
    "ProjectSummary",
    "TaskStat",
    "flag_streak",
    "project_summary",
    "recent_runs",
    "task_history",
    "task_stats",
]


@dataclass(frozen=True)
class ProjectSummary:
    """Headline numbers for a project."""

    total_runs: int
    runs_in_window: int
    failing_runs: int
    last_run: GateRun | None
    replicates: int
    replicates_reused: int
    cost_usd: float

    @property
    def pass_rate(self) -> float:
        if not self.runs_in_window:
            return 0.0
        return 1.0 - self.failing_runs / self.runs_in_window

    @property
    def reuse_rate(self) -> float:
        total = self.replicates + self.replicates_reused
        return self.replicates_reused / total if total else 0.0


@dataclass(frozen=True)
class TaskStat:
    """How one task has behaved across recent runs."""

    task_id: str
    runs: int
    flagged: int
    flips: int
    mean_replicates: float
    last_flagged: bool
    last_delta: float

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.runs if self.runs else 0.0

    @property
    def flip_rate(self) -> float:
        """Share of consecutive run pairs where the verdict changed."""
        return self.flips / (self.runs - 1) if self.runs > 1 else 0.0

    @property
    def is_flaky(self) -> bool:
        """Unstable rather than broken.

        Needs at least three runs, because one flip out of two runs is just a
        change, and a task that is flagged every single time is not flaky.
        """
        return self.runs >= 3 and self.flip_rate >= 0.25 and self.flag_rate < 0.9

    @property
    def is_chronic(self) -> bool:
        """Flagged nearly always, and not flipping. A regression nobody fixed."""
        return self.runs >= 3 and self.flag_rate >= 0.7 and self.flip_rate < 0.25

    @property
    def label(self) -> str:
        if self.is_chronic:
            return "chronic"
        if self.is_flaky:
            return "flaky"
        if self.last_flagged:
            return "flagged"
        return "stable"


def project_summary(session: Session, project: Project, window: int = 30) -> ProjectSummary:
    """Totals over the whole project, and rates over the last ``window`` runs."""
    total = session.scalar(
        select(func.count(GateRun.id)).where(GateRun.project_id == project.id)
    )
    recent = list(
        session.scalars(
            select(GateRun)
            .where(GateRun.project_id == project.id)
            .order_by(GateRun.created_at.desc())
            .limit(window)
        )
    )
    return ProjectSummary(
        total_runs=int(total or 0),
        runs_in_window=len(recent),
        failing_runs=sum(1 for r in recent if r.verdict in {"fail", "error"}),
        last_run=recent[0] if recent else None,
        replicates=sum(r.replicates_run for r in recent),
        replicates_reused=sum(r.replicates_reused for r in recent),
        cost_usd=sum(r.cost_usd for r in recent),
    )


def recent_runs(session: Session, project: Project, limit: int = 20) -> list[GateRun]:
    return list(
        session.scalars(
            select(GateRun)
            .where(GateRun.project_id == project.id)
            .order_by(GateRun.created_at.desc())
            .limit(limit)
        )
    )


def task_stats(session: Session, project: Project, window: int = 30) -> list[TaskStat]:
    """Per-task behaviour over the last ``window`` runs, worst first.

    Done in one query over the window's task rows rather than per task, because
    a suite with two hundred tasks would otherwise mean two hundred round trips
    to render one page.
    """
    run_ids = list(
        session.scalars(
            select(GateRun.id)
            .where(GateRun.project_id == project.id)
            .order_by(GateRun.created_at.desc())
            .limit(window)
        )
    )
    if not run_ids:
        return []

    rows = list(
        session.execute(
            select(
                TaskResult.task_id,
                TaskResult.flagged,
                TaskResult.replicates,
                TaskResult.delta,
                TaskResult.created_at,
            )
            .where(TaskResult.gate_run_id.in_(run_ids))
            .order_by(TaskResult.task_id, TaskResult.created_at)
        )
    )

    grouped: dict[str, list[tuple[bool, int, float, dt.datetime]]] = {}
    for task_id, flagged, replicates, delta, created_at in rows:
        grouped.setdefault(task_id, []).append((bool(flagged), replicates, delta, created_at))

    stats = []
    for task_id, entries in grouped.items():
        flags = [e[0] for e in entries]
        flips = sum(1 for a, b in pairwise(flags) if a != b)
        stats.append(
            TaskStat(
                task_id=task_id,
                runs=len(entries),
                flagged=sum(flags),
                flips=flips,
                mean_replicates=sum(e[1] for e in entries) / len(entries),
                last_flagged=flags[-1],
                last_delta=entries[-1][2],
            )
        )

    # Chronic first, then flaky, then whatever is flagged right now. Within a
    # band, the noisiest and most-flagged rise.
    def rank(s: TaskStat) -> tuple[int, float, float]:
        band = 0 if s.is_chronic else 1 if s.is_flaky else 2 if s.last_flagged else 3
        return (band, -s.flip_rate, -s.flag_rate)

    stats.sort(key=rank)
    return stats


def task_history(
    session: Session, project: Project, task_id: str, limit: int = 50
) -> list[tuple[TaskResult, GateRun]]:
    """Every recorded result for one task, newest first, with its run."""
    rows = session.execute(
        select(TaskResult, GateRun)
        .join(GateRun, TaskResult.gate_run_id == GateRun.id)
        .where(TaskResult.project_id == project.id, TaskResult.task_id == task_id)
        .order_by(GateRun.created_at.desc())
        .limit(limit)
    )
    return [(task, run) for task, run in rows]


def flag_streak(history: list[tuple[TaskResult, GateRun]]) -> int:
    """How many of the most recent runs in a row flagged this task.

    Zero means the latest run did not flag it. Used to say "flagged in the last
    four runs" instead of making the reader count rows.
    """
    streak = 0
    for task, _ in history:
        if not task.flagged:
            break
        streak += 1
    return streak
