"""Synthetic Go board generator — § 2.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GoBoardSample:
    black_count: int
    white_count: int
    occupancy: Tensor  # [H, W] 0=empty, 1=black, 2=white
    patch_labels: Tensor  # [L] binary black-at-patch


def _grid_from_board_size(board_size: int, stone_pixels: int) -> tuple[int, int]:
    side = board_size * stone_pixels
    patches = side // stone_pixels
    return side, patches


def sample_go_board(
    black_count: int,
    *,
    board_size: int = 19,
    stone_pixels: int = 14,
    distractor_delta: int = 30,
    generator: torch.Generator | None = None,
) -> GoBoardSample:
    """Place non-overlapping black/white stones; white count in [B-Δ, B+Δ]."""
    side, patches = _grid_from_board_size(board_size, stone_pixels)
    g = generator or torch.Generator()
    white_lo = max(0, black_count - distractor_delta)
    white_hi = black_count + distractor_delta
    white_count = int(torch.randint(white_lo, white_hi + 1, (1,), generator=g).item())
    max_cells = patches * patches
    n_total = min(black_count + white_count, max_cells)
    black_count = min(black_count, n_total)
    white_count = n_total - black_count

    occ = torch.zeros(patches, patches, dtype=torch.long)
    flat_idx = torch.randperm(max_cells, generator=g)[:n_total]
    black_idx = flat_idx[:black_count]
    white_idx = flat_idx[black_count:]
    occ.view(-1)[black_idx] = 1
    occ.view(-1)[white_idx] = 2

    patch_labels = (occ.view(-1) == 1).float()
    return GoBoardSample(black_count, white_count, occ, patch_labels)


def patch_embeddings_from_board(sample: GoBoardSample, *, dim: int = 32, seed: int = 0) -> Tensor:
    """Toy ViT patch outputs: linearly separable black vs background — § 3.2."""
    g = torch.Generator().manual_seed(seed + sample.black_count)
    labels = sample.patch_labels.unsqueeze(-1)
    black_vec = torch.randn(dim, generator=g)
    bg_vec = torch.randn(dim, generator=g)
    noise = torch.randn(sample.patch_labels.numel(), dim, generator=g) * 0.05
    emb = labels * black_vec + (1 - labels) * bg_vec + noise
    return emb
