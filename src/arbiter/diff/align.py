"""Aligning two agent trajectories to find where they diverged.

A flagged task tells you *that* the candidate got worse. The next question is
always *where*, and staring at two step lists side by side is a poor use of a
human. Sequence alignment answers it directly: line the two runs up, and the
first place they fail to match is where the behaviour changed.

Needleman-Wunsch rather than a diff library, because agent steps are not lines
of text. Two runs that call the same tool with different arguments should align
as a substitution, not as an unrelated insert-and-delete pair, and the scoring
function is where that judgement lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..runner.types import Step

__all__ = ["AlignOp", "Alignment", "OpKind", "align"]


class OpKind(StrEnum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    INSERT = "insert"
    DELETE = "delete"

    @property
    def symbol(self) -> str:
        return {
            OpKind.MATCH: "=",
            OpKind.SUBSTITUTE: "~",
            OpKind.INSERT: "+",
            OpKind.DELETE: "-",
        }[self]


@dataclass(frozen=True)
class AlignOp:
    """One position in the alignment.

    ``baseline`` is None for a step the candidate added, ``candidate`` is None
    for one it dropped.
    """

    kind: OpKind
    baseline: Step | None
    candidate: Step | None

    @property
    def args_changed(self) -> bool:
        if self.baseline is None or self.candidate is None:
            return False
        return self.baseline.args_digest != self.candidate.args_digest

    def describe(self) -> str:
        if self.kind is OpKind.MATCH:
            name = self.baseline.name if self.baseline else "?"
            suffix = " (different arguments)" if self.args_changed else ""
            return f"{self.kind.symbol} {name}{suffix}"
        if self.kind is OpKind.SUBSTITUTE:
            left = self.baseline.signature if self.baseline else "?"
            right = self.candidate.signature if self.candidate else "?"
            return f"{self.kind.symbol} {left} -> {right}"
        if self.kind is OpKind.INSERT:
            return f"{self.kind.symbol} {self.candidate.signature if self.candidate else '?'}"
        return f"{self.kind.symbol} {self.baseline.signature if self.baseline else '?'}"


@dataclass(frozen=True)
class Alignment:
    """The result of lining two trajectories up."""

    ops: tuple[AlignOp, ...]
    score: int

    @property
    def matches(self) -> int:
        return sum(1 for op in self.ops if op.kind is OpKind.MATCH)

    @property
    def similarity(self) -> float:
        """Matched positions as a share of the alignment length."""
        return self.matches / len(self.ops) if self.ops else 1.0

    @property
    def identical(self) -> bool:
        return all(
            op.kind is OpKind.MATCH and not op.args_changed for op in self.ops
        )

    @property
    def first_divergence(self) -> int | None:
        """Index of the first op that is not a clean match, if there is one."""
        for index, op in enumerate(self.ops):
            if op.kind is not OpKind.MATCH or op.args_changed:
                return index
        return None

    def summary(self, limit: int = 12) -> list[str]:
        """A short, readable rendering, centred on where things went wrong."""
        divergence = self.first_divergence
        if divergence is None:
            return ["trajectories are identical"]
        start = max(divergence - 2, 0)
        window = self.ops[start : start + limit]
        lines = [op.describe() for op in window]
        if start + limit < len(self.ops):
            lines.append(f"... {len(self.ops) - start - limit} more steps")
        return lines


def _score(baseline: Step, candidate: Step) -> int:
    """How well two steps correspond.

    Same tool and same outcome is a match. Same tool but one of them errored is
    a near miss and still worth aligning, because that is precisely the pattern
    a regression produces. Different tools score negative so the algorithm
    prefers a gap over a nonsense pairing.
    """
    if baseline.signature == candidate.signature:
        return 3
    if baseline.name == candidate.name and baseline.kind == candidate.kind:
        return 1
    if baseline.name == candidate.name:
        return 0
    return -2


def align(baseline: tuple[Step, ...], candidate: tuple[Step, ...], gap: int = -2) -> Alignment:
    """Globally align two step sequences with Needleman-Wunsch."""
    n, m = len(baseline), len(candidate)
    if n == 0 and m == 0:
        return Alignment(ops=(), score=0)

    # matrix[i][j] is the best score aligning the first i baseline steps with
    # the first j candidate steps.
    matrix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        matrix[i][0] = matrix[i - 1][0] + gap
    for j in range(1, m + 1):
        matrix[0][j] = matrix[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = matrix[i - 1][j - 1] + _score(baseline[i - 1], candidate[j - 1])
            up = matrix[i - 1][j] + gap
            left = matrix[i][j - 1] + gap
            matrix[i][j] = max(diagonal, up, left)

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            step_score = _score(baseline[i - 1], candidate[j - 1])
            if matrix[i][j] == matrix[i - 1][j - 1] + step_score:
                kind = (
                    OpKind.MATCH
                    if baseline[i - 1].signature == candidate[j - 1].signature
                    else OpKind.SUBSTITUTE
                )
                ops.append(AlignOp(kind, baseline[i - 1], candidate[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and matrix[i][j] == matrix[i - 1][j] + gap:
            ops.append(AlignOp(OpKind.DELETE, baseline[i - 1], None))
            i -= 1
            continue
        ops.append(AlignOp(OpKind.INSERT, None, candidate[j - 1]))
        j -= 1
    ops.reverse()
    return Alignment(ops=tuple(ops), score=matrix[n][m])
