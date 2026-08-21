"""Tile grid and polyomino construction (Sec. 4–5.1)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Polyomino:
    """Connected relevant tiles as grid positions (row, col)."""

    positions: frozenset[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def height(self) -> int:
        rows = [p[0] for p in self.positions]
        return max(rows) - min(rows) + 1 if rows else 0

    @property
    def width(self) -> int:
        cols = [p[1] for p in self.positions]
        return max(cols) - min(cols) + 1 if cols else 0

    @property
    def bbox_origin(self) -> tuple[int, int]:
        if not self.positions:
            return 0, 0
        rows = [p[0] for p in self.positions]
        cols = [p[1] for p in self.positions]
        return min(rows), min(cols)


def frame_difference(img_f: Tensor, img_prev: Tensor) -> Tensor:
    """ΔImg = |Img_f − Img_{f-1}| element-wise (Eq. 1)."""
    return (img_f - img_prev).abs()


def scores_to_relevant_mask(scores: Tensor, threshold: float) -> Tensor:
    """Boolean H×W mask where Score_{i,j} >= T_r."""
    return scores >= threshold


def connected_polyominoes(mask: Tensor) -> list[Polyomino]:
    """Group 4-connected relevant tiles into polyominoes."""
    h, w = mask.shape
    visited = torch.zeros_like(mask, dtype=torch.bool)
    polyominoes: list[Polyomino] = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            cells: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([(i, j)])
            visited[i, j] = True
            while q:
                r, c = q.popleft()
                cells.add((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            polyominoes.append(Polyomino(frozenset(cells)))
    return polyominoes
