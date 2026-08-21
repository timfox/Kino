"""EXIF metadata tokenizer for camera style matching (Sec. 3.4, Eq. 5)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.deltacam.config import DeltaCamConfig, IntrinsicRange
from ltx_trainer.deltacam.delta_params import normalize_absolute


def normalize_metadata_vector(
    metadata: Tensor | Sequence[float],
    cfg: DeltaCamConfig | None = None,
    *,
    param_keys: Sequence[str] | None = None,
) -> Tensor:
    """Map physical EXIF vector to ``[0, 1]`` per Eq. (5)."""
    cfg = cfg or DeltaCamConfig()
    keys = list(param_keys or cfg.ranges.keys())
    if isinstance(metadata, Tensor):
        vals = metadata.detach().cpu().tolist()
    else:
        vals = list(metadata)
    if len(vals) != len(keys):
        raise ValueError(f"Expected {len(keys)} metadata fields, got {len(vals)}")
    out = []
    for v, key in zip(vals, keys, strict=True):
        spec = cfg.range_for(key)
        out.append(normalize_absolute(float(v), spec))
    return torch.tensor(out, dtype=torch.float32)


class ExifMetadataTokenizer(nn.Module):
    """Φ_meta: normalized metadata → style embedding ξ_t (two-layer MLP, zero-init output)."""

    def __init__(self, num_meta: int, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_meta, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, normalized_meta: Tensor) -> Tensor:
        if normalized_meta.dim() == 1:
            normalized_meta = normalized_meta.unsqueeze(0)
        return self.net(normalized_meta)


def exif_sequence_embedding(
    metadata_traj: Tensor,
    tokenizer: ExifMetadataTokenizer,
) -> Tensor:
    """``metadata_traj``: ``[T, N_meta]`` per-frame normalized metadata → ``[T, d]`` (batched MLP)."""
    if metadata_traj.dim() != 2:
        raise ValueError(f"Expected [T, N], got {tuple(metadata_traj.shape)}")
    return tokenizer(metadata_traj)
