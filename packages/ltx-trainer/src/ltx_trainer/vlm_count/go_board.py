"""Synthetic Go-board counting stub (19×19 toy lab + 6×6 Qwen validation grid)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GoBoardSample:
    grid_size: int
    n_black: int
    n_white: int
    black_mask: np.ndarray  # (H, W) bool
    white_mask: np.ndarray


def _place_stones(
    rng: np.random.Generator,
    grid: int,
    n: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    attempts = 0
    while len(coords) < n and attempts < n * 100:
        r, c = int(rng.integers(0, grid)), int(rng.integers(0, grid))
        if (r, c) not in occupied and (r, c) not in coords:
            coords.add((r, c))
        attempts += 1
    return coords


def sample_go_board(
    n_black: int,
    *,
    grid_size: int = 19,
    distractor_delta: int = 0,
    rng: np.random.Generator | None = None,
) -> GoBoardSample:
    """Black = target; white = distractors with |n_white - n_black| <= distractor_delta (±30 toy, ±5 Qwen)."""
    rng = rng or np.random.default_rng()
    n_white = max(0, n_black + int(rng.integers(-distractor_delta, distractor_delta + 1)))
    black = _place_stones(rng, grid_size, n_black, set())
    white = _place_stones(rng, grid_size, n_white, black)
    mask_b = np.zeros((grid_size, grid_size), dtype=bool)
    mask_w = np.zeros((grid_size, grid_size), dtype=bool)
    for r, c in black:
        mask_b[r, c] = True
    for r, c in white:
        mask_w[r, c] = True
    return GoBoardSample(
        grid_size=grid_size,
        n_black=n_black,
        n_white=n_white,
        black_mask=mask_b,
        white_mask=mask_w,
    )


def patch_embeddings_from_board(
    sample: GoBoardSample,
    *,
    patch_size: int = 14,
    embed_dim: int = 32,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Toy patch tokens: one embedding per patch; label 1 if patch center has black stone."""
    rng = rng or np.random.default_rng(0)
    g = sample.grid_size
    n_ph = max(1, g // patch_size)
    n_pw = max(1, g // patch_size)
    emb = rng.standard_normal((n_ph * n_pw, embed_dim))
    lab = np.zeros(n_ph * n_pw, dtype=np.float64)
    for pi in range(n_ph):
        for pj in range(n_pw):
            idx = pi * n_pw + pj
            r, c = min(pi * patch_size, g - 1), min(pj * patch_size, g - 1)
            if sample.black_mask[r, c]:
                lab[idx] = 1.0
                emb[idx] += 2.0  # separable signal for probe stub
    return emb, lab
