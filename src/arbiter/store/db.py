"""SQLite-backed run store.

The store exists for one reason above all others: **baseline runs are expensive
and reusable**. A pull request changes the candidate, not the baseline, so the
baseline's replicate at seed 7 on task ``refund_flow`` is the same run it was
yesterday. Caching those is typically a bigger saving than the sequential
testing is, and it costs a table with a unique index.

Reuse is keyed on the variant hash from
:meth:`~arbiter.config.SuiteConfig.variant_id`, so changing the model, the
prompt or the target command silently invalidates the cache instead of quietly
comparing against a build that no longer exists.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from ..runner.types import RunOutcome

__all__ = ["SCHEMA_VERSION", "Store"]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    variant_id  TEXT    NOT NULL,
    suite       TEXT    NOT NULL,
    task_id     TEXT    NOT NULL,
    replicate   INTEGER NOT NULL,
    seed        INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    score       REAL,
    error       TEXT,
    cost_usd    REAL    NOT NULL DEFAULT 0,
    latency_ms  REAL    NOT NULL DEFAULT 0,
    payload     TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    UNIQUE (variant_id, task_id, replicate)
);

CREATE INDEX IF NOT EXISTS runs_lookup ON runs (variant_id, task_id);

CREATE TABLE IF NOT EXISTS gates (
    id                INTEGER PRIMARY KEY,
    suite             TEXT    NOT NULL,
    baseline_variant  TEXT    NOT NULL,
    candidate_variant TEXT    NOT NULL,
    verdict           TEXT    NOT NULL,
    n_tasks           INTEGER NOT NULL,
    n_flagged         INTEGER NOT NULL,
    replicates_run    INTEGER NOT NULL,
    replicates_reused INTEGER NOT NULL,
    cost_usd          REAL    NOT NULL,
    wall_seconds      REAL    NOT NULL,
    summary           TEXT    NOT NULL,
    created_at        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    id          INTEGER PRIMARY KEY,
    gate_id     INTEGER NOT NULL REFERENCES gates (id) ON DELETE CASCADE,
    task_id     TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    flagged     INTEGER NOT NULL,
    replicates  INTEGER NOT NULL,
    e_value     REAL    NOT NULL,
    anytime_p   REAL    NOT NULL,
    adjusted_p  REAL    NOT NULL,
    delta       REAL    NOT NULL,
    detail      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS verdicts_by_gate ON verdicts (gate_id);
"""


class Store:
    """Thin, synchronous wrapper over the SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- runs ---------------------------------------------------------------

    def record_run(
        self,
        *,
        variant_id: str,
        suite: str,
        task_id: str,
        replicate: int,
        seed: int,
        outcome: RunOutcome,
    ) -> None:
        """Persist one replicate. Re-recording the same cell is a no-op.

        The cell is keyed by (variant, task, replicate) rather than by a run id,
        which is what makes the cache idempotent: re-running a gate after a
        crash resumes rather than double-counting.
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO runs
                (variant_id, suite, task_id, replicate, seed, passed, score,
                 error, cost_usd, latency_ms, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                variant_id,
                suite,
                task_id,
                replicate,
                seed,
                int(outcome.passed),
                outcome.score,
                outcome.error,
                outcome.cost_usd,
                outcome.latency_ms,
                json.dumps(outcome.to_dict()),
                time.time(),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def load_runs(self, variant_id: str, task_ids: Iterable[str] | None = None) -> dict[
        tuple[str, int], RunOutcome
    ]:
        """Every stored replicate for a variant, keyed by (task_id, replicate)."""
        query = "SELECT task_id, replicate, payload FROM runs WHERE variant_id = ?"
        params: list[Any] = [variant_id]
        ids = list(task_ids) if task_ids is not None else None
        if ids:
            query += f" AND task_id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        with closing(self.conn.execute(query, params)) as cur:
            return {
                (row["task_id"], row["replicate"]): RunOutcome.from_dict(json.loads(row["payload"]))
                for row in cur
            }

    def load_task_runs(self, variant_id: str, task_id: str) -> list[tuple[int, RunOutcome]]:
        """Replicates for one task, in replicate order."""
        with closing(
            self.conn.execute(
                "SELECT replicate, payload FROM runs WHERE variant_id = ? AND task_id = ? "
                "ORDER BY replicate",
                (variant_id, task_id),
            )
        ) as cur:
            return [
                (row["replicate"], RunOutcome.from_dict(json.loads(row["payload"]))) for row in cur
            ]

    def count_runs(self, variant_id: str) -> int:
        with closing(
            self.conn.execute("SELECT COUNT(*) AS n FROM runs WHERE variant_id = ?", (variant_id,))
        ) as cur:
            return int(cur.fetchone()["n"])

    def prune_variant(self, variant_id: str) -> int:
        """Drop every stored run for a variant. Returns rows removed."""
        cur = self.conn.execute("DELETE FROM runs WHERE variant_id = ?", (variant_id,))
        self.conn.commit()
        return cur.rowcount

    # -- gates --------------------------------------------------------------

    def record_gate(self, summary: dict[str, Any]) -> int:
        """Persist a gate result and its per-task verdicts. Returns the gate id."""
        cur = self.conn.execute(
            """
            INSERT INTO gates
                (suite, baseline_variant, candidate_variant, verdict, n_tasks,
                 n_flagged, replicates_run, replicates_reused, cost_usd,
                 wall_seconds, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["suite"],
                summary["baseline_variant"],
                summary["candidate_variant"],
                summary["verdict"],
                summary["n_tasks"],
                summary["n_flagged"],
                summary["replicates_run"],
                summary["replicates_reused"],
                summary["cost_usd"],
                summary["wall_seconds"],
                json.dumps(summary),
                time.time(),
            ),
        )
        gate_id = int(cur.lastrowid or 0)
        self.conn.executemany(
            """
            INSERT INTO verdicts
                (gate_id, task_id, verdict, flagged, replicates, e_value,
                 anytime_p, adjusted_p, delta, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    gate_id,
                    t["task_id"],
                    t["verdict"],
                    int(t["flagged"]),
                    t["replicates"],
                    t["e_value"],
                    t["anytime_p"],
                    t["adjusted_p"],
                    t["delta"],
                    json.dumps(t),
                )
                for t in summary["tasks"]
            ],
        )
        self.conn.commit()
        return gate_id

    def recent_gates(self, suite: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT * FROM gates"
        params: list[Any] = []
        if suite:
            query += " WHERE suite = ?"
            params.append(suite)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with closing(self.conn.execute(query, params)) as cur:
            return [dict(row) for row in cur]

    def gate_summary(self, gate_id: int) -> dict[str, Any] | None:
        with closing(
            self.conn.execute("SELECT summary FROM gates WHERE id = ?", (gate_id,))
        ) as cur:
            row = cur.fetchone()
        return json.loads(row["summary"]) if row else None
