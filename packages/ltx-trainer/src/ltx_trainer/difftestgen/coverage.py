"""Union coverage and behavioral difference detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageCounts:
    old_changed: int
    new_changed: int
    old_covered: int
    new_covered: int

    @property
    def union_coverage(self) -> float:
        """Definition 3: (covered_old + covered_new) / (changed_old + changed_new)."""
        denom = self.old_changed + self.new_changed
        if denom <= 0:
            return 1.0
        return (self.old_covered + self.new_covered) / denom


def union_coverage(
    *,
    old_changed: int,
    new_changed: int,
    old_covered: int,
    new_covered: int,
) -> float:
    return CoverageCounts(old_changed, new_changed, old_covered, new_covered).union_coverage


@dataclass
class ExecResult:
    output: str | None = None
    error_type: str | None = None


def has_behavioral_difference(old: ExecResult, new: ExecResult) -> bool:
    """Definition 2: error-type mismatch, one-sided error, or output mismatch."""
    if old.error_type and new.error_type:
        return old.error_type != new.error_type
    if bool(old.error_type) != bool(new.error_type):
        return True
    if old.error_type is None and new.error_type is None:
        return (old.output or "") != (new.output or "")
    return False


def annotate_coverage(lines: list[str], covered: set[int]) -> list[str]:
    """Encode # COVERED / # TO_COVER comments for LLM feedback (§II-F)."""
    out = []
    for i, line in enumerate(lines, start=1):
        tag = " # COVERED" if i in covered else " # TO_COVER"
        if line.strip() and not line.strip().startswith("#"):
            out.append(line.rstrip() + tag)
        else:
            out.append(line)
    return out
