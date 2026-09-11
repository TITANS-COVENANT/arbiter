"""A synthetic agent with known ground truth, and the harness that measures against it."""

from .agent import SimSpec, baseline, candidate, simulate_run
from .harness import (
    ExperimentResult,
    ScenarioSpec,
    TrialOutcome,
    make_suite,
    run_experiment,
)

__all__ = [
    "ExperimentResult",
    "ScenarioSpec",
    "SimSpec",
    "TrialOutcome",
    "baseline",
    "candidate",
    "make_suite",
    "run_experiment",
    "simulate_run",
]
