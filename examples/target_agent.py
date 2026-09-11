"""A worked example of an eval target: a small retrieval agent with a real bug.

Runnable with no API key and no network, so the example suite works out of the
box, but it is shaped like a real target rather than like a mock. It does a tool
loop, it uses the seed for every random decision it makes, and the candidate
build contains a specific plausible mistake.

The bug: the candidate truncates retrieved context to one passage. Questions
needing two facts to answer now fail almost every time, while questions needing
one are mostly fine. That asymmetry is exactly what a per-task gate should find
and a suite-wide average would bury.

There is a third case, and it is the interesting one. ``current_ceo`` needs only
a single fact, so it was meant to be unaffected. But two passages in the corpus
tie on retrieval score for that question, and only one of them carries the
answer, so truncating to one passage breaks it about half the time. arbiter flags
it. That was not planted deliberately; it is what happens when you test per task
instead of averaging, and it is a fair illustration of why the per-task verdict is
worth the extra machinery.

Usage, which is what ``examples/suite.yaml`` invokes:

    echo '{"task": {...}, "seed": 7}' | python examples/target_agent.py --build candidate
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

# A tiny corpus. Each passage is one fact.
CORPUS: dict[str, str] = {
    "founded": "Northwind was founded in 1994 in Leeds.",
    "ceo": "Northwind's chief executive is Dara Okonjo.",
    "ceo_start": "Dara Okonjo became chief executive in 2019.",
    "revenue": "Northwind reported revenue of 41 million pounds in 2023.",
    "staff": "Northwind employs 320 people.",
    "hq": "Northwind moved its head office to Manchester in 2021.",
    "product": "Northwind's main product is a warehouse routing system.",
}


def retrieve(query_terms: list[str], rng: random.Random, limit: int) -> list[str]:
    """Score passages by term overlap, with a seed-driven tie-break.

    The tie-break is the point. Real retrievers are not deterministic under
    load, and using the seed for it is what lets the two builds be compared on
    equal footing.
    """
    scored = []
    for key, passage in CORPUS.items():
        words = set(passage.lower().replace(".", "").replace(",", "").split())
        overlap = len(words & {t.lower() for t in query_terms})
        if overlap:
            scored.append((overlap, rng.random(), key, passage))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [passage for _, _, _, passage in scored[:limit]]


def run(task_input: dict[str, Any], seed: int, build: str) -> dict[str, Any]:
    """Answer one question and report what happened."""
    rng = random.Random(seed)
    question: str = task_input["question"]
    needs: list[str] = task_input["needs"]

    # Two sources of randomness, and the split between them is the point.
    #
    # `rng` is seeded from the seed alone, so both builds make identical
    # retrieval decisions on a given replicate. That is what the seed is for.
    #
    # `wobble` is seeded from the seed *and the build*, so it differs between
    # them. Real model serving is not reproducible even at a fixed seed, and a
    # target that pretended otherwise would make the gate look better than it is.
    wobble = random.Random(f"{seed}:{build}")

    steps: list[dict[str, Any]] = [
        {"kind": "tool_call", "name": "plan", "ok": True, "args": {"question": question}}
    ]

    terms = [w.strip("?,.").lower() for w in question.split() if len(w) > 3]

    # The bug lives here.
    limit = 1 if build == "candidate" else 3
    passages = retrieve(terms, rng, limit)
    steps.append(
        {
            "kind": "tool_call",
            "name": "search",
            "ok": bool(passages),
            "args": {"terms": terms, "limit": limit},
        }
    )

    if not passages:
        steps.append(
            {"kind": "error", "name": "no_results", "ok": False, "detail": "search found nothing"}
        )
        return {
            "passed": False,
            "error": "search found nothing",
            "steps": steps,
            "cost_usd": 0.004,
        }

    context = " ".join(passages)
    steps.append({"kind": "tool_call", "name": "read", "ok": True, "args": {"n": len(passages)}})

    missing = [fact for fact in needs if CORPUS[fact] not in context]
    if missing:
        steps.append(
            {
                "kind": "error",
                "name": "incomplete_context",
                "ok": False,
                "detail": f"answer needs {len(needs)} facts but context had "
                f"{len(needs) - len(missing)}",
            }
        )
        return {
            "passed": False,
            "error": f"missing supporting fact: {missing[0]}",
            "steps": steps,
            "cost_usd": 0.008,
        }

    # Even with the right context a model sometimes gets it wrong, and it does so
    # independently of the other build.
    if wobble.random() < 0.06:
        steps.append(
            {"kind": "error", "name": "wrong_answer", "ok": False, "detail": "misread the figure"}
        )
        return {
            "passed": False,
            "error": "answer did not match expected value",
            "steps": steps,
            "cost_usd": 0.011,
        }

    steps.append({"kind": "message", "name": "answer", "ok": True, "args": {}})
    return {"passed": True, "steps": steps, "cost_usd": 0.011, "score": 1.0}


def baseline(task_input: dict[str, Any], seed: int, **_: Any) -> dict[str, Any]:
    """In-process entry point for the baseline build."""
    return run(task_input, seed, "baseline")


def candidate(task_input: dict[str, Any], seed: int, **_: Any) -> dict[str, Any]:
    """In-process entry point for the candidate build."""
    return run(task_input, seed, "candidate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", choices=["baseline", "candidate"], required=True)
    args = parser.parse_args()
    payload = json.loads(sys.stdin.read())
    result = run(payload["task"]["input"], int(payload["seed"]), args.build)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
