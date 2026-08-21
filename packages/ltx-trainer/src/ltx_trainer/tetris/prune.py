"""Polyomino pruning under maximum tile sampling gaps (Sec. 5.2)."""

from __future__ import annotations

from ltx_trainer.tetris.tiles import Polyomino


def _is_covered(
    frames: list[list[Polyomino]],
    selected: list[set[int]],
    f_start: int,
    f_end: int,
    i: int,
    j: int,
) -> bool:
    for f in range(f_start, f_end):
        for k in selected[f]:
            if (i, j) in frames[f][k].positions:
                return True
    return False


def prune_polyominoes(
    frames: list[list[Polyomino]],
    gap_matrix: list[list[int]],
) -> list[list[Polyomino]]:
    """Greedy ILP approximation: drop polyominoes while gap coverage holds."""
    n_frames = len(frames)
    if n_frames == 0:
        return []
    h = len(gap_matrix)
    w = len(gap_matrix[0]) if h else 0
    selected = [set(range(len(p))) for p in frames]

    for f in range(n_frames):
        order = sorted(selected[f], key=lambda k: len(frames[f][k]))
        for k in order:
            trial = [s.copy() for s in selected]
            trial[f].discard(k)
            ok = True
            for i in range(h):
                for j in range(w):
                    g = gap_matrix[i][j]
                    for f0 in range(n_frames):
                        f1 = min(f0 + g, n_frames)
                        needs = any(
                            (i, j) in frames[ff][kk].positions
                            for ff in range(f0, f1)
                            for kk in selected[ff]
                        )
                        if needs and not _is_covered(frames, trial, f0, f1, i, j):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                selected = trial

    return [[frames[f][k] for k in sorted(selected[f])] for f in range(n_frames)]
