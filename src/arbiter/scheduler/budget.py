"""Budget tracking.

A gate run is allowed to spend replicates, dollars and wall-clock seconds.
Whichever ceiling binds first stops the run, and everything still undecided at
that moment is reported inconclusive. Silently passing undecided tasks is the
failure mode that makes people stop believing eval gates, so it is not an option
here: an underfunded gate is supposed to look underfunded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import BudgetConfig

__all__ = ["BudgetTracker"]


@dataclass
class BudgetTracker:
    """Running spend against the configured ceilings."""

    config: BudgetConfig
    replicates: int = 0
    cost_usd: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def charge(self, *, replicates: int = 0, cost_usd: float = 0.0) -> None:
        self.replicates += replicates
        self.cost_usd += cost_usd

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def exhausted(self) -> bool:
        return self.binding_constraint is not None

    @property
    def binding_constraint(self) -> str | None:
        """Which ceiling stopped the run, if any. Useful in the report."""
        cfg = self.config
        if cfg.max_replicates is not None and self.replicates >= cfg.max_replicates:
            return "max_replicates"
        if cfg.max_cost_usd is not None and self.cost_usd >= cfg.max_cost_usd:
            return "max_cost_usd"
        if cfg.max_seconds is not None and self.elapsed_seconds >= cfg.max_seconds:
            return "max_seconds"
        return None

    def headroom(self) -> int:
        """How many more replicates the budget will admit right now.

        Cost headroom is converted using the observed average cost per
        replicate, so a suite that turns out to be pricier than expected slows
        down instead of blowing through the ceiling in one batch.
        """
        cfg = self.config
        limits = [cfg.batch_size]
        if cfg.max_replicates is not None:
            limits.append(max(cfg.max_replicates - self.replicates, 0))
        if cfg.max_cost_usd is not None and self.replicates > 0:
            per_replicate = self.cost_usd / self.replicates
            if per_replicate > 0:
                limits.append(int(max(cfg.max_cost_usd - self.cost_usd, 0.0) / per_replicate))
        return max(min(limits), 0)

    def snapshot(self) -> dict[str, object]:
        return {
            "replicates": self.replicates,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "binding_constraint": self.binding_constraint,
        }
