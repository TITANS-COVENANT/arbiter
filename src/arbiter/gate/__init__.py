"""The gate: orchestration, per-task decisions, and the final verdict."""

from .decide import GateResult, evidence_threshold, gate_sync, run_gate
from .task_test import StopReason, TaskResult, TaskTest

__all__ = [
    "GateResult",
    "StopReason",
    "TaskResult",
    "TaskTest",
    "evidence_threshold",
    "gate_sync",
    "run_gate",
]
