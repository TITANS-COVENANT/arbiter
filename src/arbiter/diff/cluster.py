"""Grouping failures by how they failed.

Forty failing runs are not forty problems. They are usually two or three
problems with different row ids in the message. Normalising the volatile parts
out of an error string and grouping on what is left turns a wall of logs into
"thirty-one of these are the same timeout, and here is the one that is new".

The normalisation is deliberately crude. Anything clever enough to parse real
error formats is clever enough to be wrong about an error format it has not
seen, and the failure mode of a too-clever normaliser is silently merging two
distinct bugs.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..runner.types import RunOutcome

__all__ = ["FailureMode", "ModeDelta", "cluster_failures", "compare_modes", "normalise_error"]

# Order matters. The path rule would otherwise eat the tail of a URL and leave
# "https:/<path>" behind, so URLs are matched first.
_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[a-zA-Z][\w+.-]*://[^\s'\"]+"), "<url>"),
    (re.compile(r"[A-Za-z]:\\[^\s'\"]+|/(?:[\w.-]+/)+[\w.-]+"), "<path>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<hex>"),
    (re.compile(r"\b\d+\.\d+\b"), "<float>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"'[^']*'|\"[^\"]*\""), "<str>"),
    (re.compile(r"\s+"), " "),
)


def normalise_error(text: str | None, max_length: int = 160) -> str:
    """Strip the parts of an error message that change run to run."""
    if not text:
        return "<no message>"
    normalised = text.strip()
    for pattern, replacement in _SUBSTITUTIONS:
        normalised = pattern.sub(replacement, normalised)
    normalised = normalised.strip()
    return normalised[:max_length] if normalised else "<no message>"


@dataclass
class FailureMode:
    """A group of failures that look like the same problem."""

    signature: str
    count: int = 0
    step_signature: str = ""
    examples: list[str] = field(default_factory=list)

    def add(self, raw: str | None) -> None:
        self.count += 1
        if raw and len(self.examples) < 3 and raw not in self.examples:
            self.examples.append(raw)


def cluster_failures(outcomes: Iterable[RunOutcome]) -> list[FailureMode]:
    """Group failing runs by normalised error, most common first.

    The last step's signature is folded into the key, so a wrong answer produced
    after a search and one produced after a retry loop stay separate even when
    the message is identical.
    """
    modes: dict[str, FailureMode] = {}
    for outcome in outcomes:
        if outcome.passed:
            continue
        step_signature = outcome.steps[-1].signature if outcome.steps else ""
        key = f"{step_signature}|{normalise_error(outcome.error)}"
        mode = modes.get(key)
        if mode is None:
            mode = FailureMode(
                signature=normalise_error(outcome.error), step_signature=step_signature
            )
            modes[key] = mode
        mode.add(outcome.error)
    return sorted(modes.values(), key=lambda m: (-m.count, m.signature))


@dataclass(frozen=True)
class ModeDelta:
    """How often a failure mode occurs in each build."""

    signature: str
    step_signature: str
    baseline_count: int
    candidate_count: int
    example: str

    @property
    def delta(self) -> int:
        return self.candidate_count - self.baseline_count

    @property
    def is_new(self) -> bool:
        """A way of failing the candidate invented.

        Usually the single most useful line in the whole report: it is the
        difference between "the same flakiness, slightly more of it" and "this
        change broke something specific".
        """
        return self.baseline_count == 0 and self.candidate_count > 0

    @property
    def is_fixed(self) -> bool:
        return self.candidate_count == 0 and self.baseline_count > 0


def compare_modes(
    baseline: Sequence[RunOutcome], candidate: Sequence[RunOutcome]
) -> list[ModeDelta]:
    """Failure modes of both builds, ordered by how much worse each one got."""
    base_modes = {f"{m.step_signature}|{m.signature}": m for m in cluster_failures(baseline)}
    cand_modes = {f"{m.step_signature}|{m.signature}": m for m in cluster_failures(candidate)}
    deltas = []
    for key in dict.fromkeys([*cand_modes, *base_modes]):
        mode = cand_modes.get(key) or base_modes[key]
        example = ""
        for source in (cand_modes.get(key), base_modes.get(key)):
            if source and source.examples:
                example = source.examples[0]
                break
        deltas.append(
            ModeDelta(
                signature=mode.signature,
                step_signature=mode.step_signature,
                baseline_count=base_modes[key].count if key in base_modes else 0,
                candidate_count=cand_modes[key].count if key in cand_modes else 0,
                example=example,
            )
        )
    return sorted(deltas, key=lambda d: (-d.delta, d.signature))


def step_name_counts(outcomes: Iterable[RunOutcome]) -> Counter[str]:
    """How often each tool was called across a set of runs."""
    counter: Counter[str] = Counter()
    for outcome in outcomes:
        for step in outcome.steps:
            counter[step.name] += 1
    return counter
