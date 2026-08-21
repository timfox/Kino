"""Sum of Absolute Differences for block motion estimation (Eq. 1)."""

from __future__ import annotations

from typing import Sequence


def sad_block(current: Sequence[int], reference: Sequence[int]) -> float:
    r"""SAD = sum_{i,j} |B_c(i,j) - B_r(i,j)| for flattened blocks."""
    if len(current) != len(reference):
        raise ValueError("blocks must have equal length")
    return float(sum(abs(int(c) - int(r)) for c, r in zip(current, reference, strict=True)))


def toy_blocks() -> tuple[list[int], list[int]]:
    """2×2 example from Eq. (2) in the paper."""
    bc = [120, 118, 115, 117]
    br = [122, 116, 114, 119]
    return bc, br
