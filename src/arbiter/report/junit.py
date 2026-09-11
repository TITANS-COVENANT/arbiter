"""JUnit XML, because every CI system on earth already knows how to read it.

One test case per eval task. A flagged task is a failure; a task the budget cut
short is skipped, not passed, so that a truncated run shows up as truncated in
whatever dashboard is consuming this rather than quietly reading as green.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from ..gate.decide import GateResult
from ..gate.task_test import StopReason

__all__ = ["render_junit"]


def render_junit(result: GateResult) -> str:
    """Serialise a gate result as a JUnit XML document."""
    flagged = [t for t in result.tasks if t.flagged]
    skipped = [t for t in result.tasks if t.stop_reason is StopReason.BUDGET]
    suite = ET.Element(
        "testsuite",
        {
            "name": f"arbiter.{result.suite}",
            "tests": str(len(result.tasks)),
            "failures": str(len(flagged)),
            "skipped": str(len(skipped)),
            "errors": "1" if result.verdict == "error" else "0",
            "time": f"{result.wall_seconds:.3f}",
        },
    )
    properties = ET.SubElement(suite, "properties")
    for key, value in (
        ("correction", result.correction),
        ("alpha", str(result.alpha)),
        ("evidence_threshold", f"{result.evidence_threshold:.1f}"),
        ("replicates_run", str(result.replicates_run)),
        ("replicates_reused", str(result.replicates_reused)),
        ("cost_usd", f"{result.cost_usd:.4f}"),
        ("baseline_variant", result.baseline_variant),
        ("candidate_variant", result.candidate_variant),
    ):
        ET.SubElement(properties, "property", {"name": key, "value": value})

    for task in result.tasks:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": f"arbiter.{result.suite}",
                "name": task.task_id,
                "time": "0",
            },
        )
        if task.flagged:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "regression",
                    "message": (
                        f"pass rate {task.baseline_rate:.0%} -> {task.candidate_rate:.0%} "
                        f"over {task.replicates} paired replicates "
                        f"(adjusted p={task.adjusted_p:.4g})"
                    ),
                },
            )
            failure.text = (
                f"{task.regressions} replicates broke and {task.improvements} were fixed at "
                f"matched seeds. e-value {task.e_value:,.1f} against a threshold of "
                f"{result.evidence_threshold:,.0f}."
            )
        elif task.stop_reason is StopReason.BUDGET:
            ET.SubElement(
                case,
                "skipped",
                {"message": "budget exhausted before this task decided"},
            )

    if result.notes:
        ET.SubElement(suite, "system-out").text = "\n".join(result.notes)
    return ET.tostring(suite, encoding="unicode", xml_declaration=True)
