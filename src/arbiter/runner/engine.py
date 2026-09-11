"""Concurrent execution of eval replicates, with the boring parts done properly.

Rate limits, retries, timeouts and a hard separation between "the task failed"
and "the harness failed". None of it is clever, all of it is the difference
between a gate you trust at 3am and one you start passing `--force` to.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import TargetError
from .adapters import Target, build_target
from .types import RunOutcome, Task

if TYPE_CHECKING:  # see adapters.py
    from ..config import TargetConfig

__all__ = ["Cell", "CellResult", "Engine", "EngineStats", "TokenBucket"]


@dataclass(frozen=True)
class Cell:
    """One unit of work: run this task at this replicate index."""

    task: Task
    replicate: int
    seed: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.task.id, self.replicate)


@dataclass(frozen=True)
class CellResult:
    """What came back, whether or not it was usable."""

    cell: Cell
    outcome: RunOutcome | None
    error: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.outcome is not None


class TokenBucket:
    """Async token bucket, for targets behind a requests-per-second limit."""

    def __init__(self, rate_per_s: float, capacity: float | None = None) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        self.rate = rate_per_s
        self.capacity = capacity if capacity is not None else max(rate_per_s, 1.0)
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            await asyncio.sleep(wait)


@dataclass
class EngineStats:
    """Running totals for one build's execution."""

    runs: int = 0
    infra_errors: int = 0
    retries: int = 0
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def infra_error_rate(self) -> float:
        attempted = self.runs + self.infra_errors
        return self.infra_errors / attempted if attempted else 0.0

    def record(self, result: CellResult) -> None:
        self.retries += max(result.attempts - 1, 0)
        if result.outcome is None:
            self.infra_errors += 1
            if result.error and len(self.errors) < 50:
                self.errors.append(result.error)
            return
        self.runs += 1
        self.cost_usd += result.outcome.cost_usd
        self.latency_ms_total += result.outcome.latency_ms


class Engine:
    """Runs cells against one target, respecting concurrency and rate limits."""

    def __init__(
        self,
        cfg: TargetConfig,
        *,
        target: Target | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg = cfg
        self.target = target if target is not None else build_target(cfg)
        self.stats = EngineStats()
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._bucket = TokenBucket(cfg.rate_limit_per_s) if cfg.rate_limit_per_s else None
        self._rng = rng or random.Random(0xA2B17E2)

    async def run_cell(self, cell: Cell) -> CellResult:
        """Run one cell, retrying transient target failures with jittered backoff."""
        last_error = "unknown"
        attempts = 0
        for attempt in range(self.cfg.max_retries + 1):
            attempts = attempt + 1
            try:
                async with self._sem:
                    if self._bucket is not None:
                        await self._bucket.acquire()
                    started = time.monotonic()
                    outcome = await asyncio.wait_for(
                        self.target.run(cell.task, cell.seed), timeout=self.cfg.timeout_s
                    )
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if outcome.latency_ms == 0.0:
                    outcome = RunOutcome(
                        passed=outcome.passed,
                        score=outcome.score,
                        error=outcome.error,
                        cost_usd=outcome.cost_usd,
                        latency_ms=elapsed_ms,
                        steps=outcome.steps,
                        metadata=outcome.metadata,
                    )
                return CellResult(cell=cell, outcome=outcome, attempts=attempts)
            except TimeoutError:
                last_error = f"timeout after {self.cfg.timeout_s}s"
            except TargetError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries:
                # Full jitter: sleep uniformly in [0, base * 2^attempt].
                backoff = min(0.25 * (2.0**attempt), 8.0)
                await asyncio.sleep(self._rng.uniform(0.0, backoff))
        return CellResult(cell=cell, outcome=None, error=last_error, attempts=attempts)

    async def run_cells(self, cells: Sequence[Cell] | Iterable[Cell]) -> list[CellResult]:
        """Run a batch concurrently and record stats in submission order."""
        cell_list = list(cells)
        if not cell_list:
            return []
        results = await asyncio.gather(*(self.run_cell(cell) for cell in cell_list))
        for result in results:
            self.stats.record(result)
        return list(results)

    async def aclose(self) -> None:
        await self.target.aclose()
