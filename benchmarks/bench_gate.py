"""Measure the gate against a simulated agent whose truth is known.

Every number in the README comes from this script. Run it yourself:

    python benchmarks/bench_gate.py --out benchmarks/results.json

It takes about ten minutes on a laptop and needs no API keys, because the agent
under test is the one in :mod:`arbiter.sim`.

Four questions get answered:

1. On a build where nothing changed, how often does the gate flag something?
   This is the claim that matters most, because a gate that cries wolf gets
   turned off, and it is the claim e-BH is supposed to underwrite.
2. On a build with real regressions, how many does it catch?
3. What did that cost against a fixed-sample design at the same error rates?
4. How much of the answer comes from the allocator and from seeding, as opposed
   to from the sequential test alone?
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from arbiter.sim.harness import ScenarioSpec, run_experiment

BASE = {
    "n_tasks": 25,
    "baseline_rate": 0.90,
    "regressed_rate": 0.70,
    "coupling": 0.7,
    "max_replicates": 150,
    "min_replicates": 4,
    "batch_size": 32,
}


def _spec(**overrides: Any) -> ScenarioSpec:
    return ScenarioSpec(**{**BASE, **overrides})


def render_table(path: str) -> str:
    """Render a saved results file as the Markdown table the README carries.

    Kept here rather than in a separate script so the published numbers are
    always a mechanical transform of the measurements, never retyped.
    """
    with open(path, encoding="utf-8") as fh:
        results = json.load(fh)
    rows = [
        "| scenario | trials | FDR | clean builds falsely flagged | regressions caught "
        "| replicates | vs. fixed sample |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in results["scenarios"].values():
        caught = (
            f"{record['per_task_power']:.0%} of tasks, {record['detection_rate']:.0%} of builds"
            if record["n_regressed"]
            else "n/a"
        )
        rows.append(
            f"| {record['description']} "
            f"| {record['n_trials']} "
            f"| {record['fdr']:.3f} "
            f"| {record['any_false_flag_rate']:.1%} "
            f"| {caught} "
            f"| {record['mean_replicates']:,.0f} "
            f"| {record['savings']:+.0%} |"
        )
    return "\n".join(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/results.json")
    parser.add_argument("--scale", type=float, default=1.0, help="scale all trial counts")
    parser.add_argument(
        "--table",
        metavar="RESULTS_JSON",
        help="render a saved results file as Markdown and exit, without measuring anything",
    )
    args = parser.parse_args()

    if args.table:
        print(render_table(args.table))
        return

    def trials(n: int) -> int:
        return max(5, int(n * args.scale))

    plan: list[tuple[str, str, ScenarioSpec, int]] = [
        (
            "null",
            "Nothing changed between the builds",
            _spec(n_regressed=0),
            trials(200),
        ),
        (
            "one_regression",
            "One task in twenty-five got worse",
            _spec(n_regressed=1),
            trials(120),
        ),
        (
            "several_regressions",
            "Three tasks in twenty-five got worse",
            _spec(n_regressed=3),
            trials(120),
        ),
        (
            "subtle_regression",
            "Three tasks slipped from 0.90 to 0.80, half the usual effect",
            _spec(n_regressed=3, regressed_rate=0.80),
            trials(80),
        ),
        (
            "alloc_round_robin",
            "Allocator: equal spend across every task",
            _spec(n_regressed=3, allocator="round-robin"),
            trials(60),
        ),
        (
            "alloc_cheapest",
            "Allocator: spend where a decision is closest",
            _spec(n_regressed=3, allocator="cheapest-to-close"),
            trials(60),
        ),
        (
            "alloc_halving",
            "Allocator: successive halving",
            _spec(n_regressed=3, allocator="successive-halving"),
            trials(60),
        ),
        (
            "coupling_none",
            "Seeds ignored by the target: no coupling between builds",
            _spec(n_regressed=3, coupling=0.0),
            trials(60),
        ),
        (
            "coupling_high",
            "Target honours seeds tightly: 90% coupling",
            _spec(n_regressed=3, coupling=0.9),
            trials(60),
        ),
        (
            "correction_none",
            "No suite-level correction, nothing changed between builds",
            _spec(n_regressed=0, correction="none"),
            trials(120),
        ),
        (
            "correction_bh",
            "Benjamini-Hochberg on anytime p-values, nothing changed",
            _spec(n_regressed=0, correction="bh"),
            trials(120),
        ),
    ]

    results: dict[str, Any] = {"generated_at": time.time(), "scenarios": {}}
    total_started = time.time()
    for key, description, spec, n_trials in plan:
        started = time.time()
        experiment = await run_experiment(spec, n_trials=n_trials, salt_prefix=key, concurrency=4)
        record = experiment.to_dict()
        record["description"] = description
        record["seconds"] = time.time() - started
        results["scenarios"][key] = record
        print(
            f"{key:24s} trials={n_trials:4d} "
            f"fdr={record['fdr']:.3f} "
            f"any_false_flag={record['any_false_flag_rate']:.3f} "
            f"detect={record['detection_rate']:.3f} "
            f"power={record['per_task_power']:.3f} "
            f"reps={record['mean_replicates']:.0f}/{record['fixed_replicates']} "
            f"({record['savings']:+.1%}) "
            f"[{record['seconds']:.0f}s]",
            flush=True,
        )
    results["total_seconds"] = time.time() - total_started

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out} in {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
