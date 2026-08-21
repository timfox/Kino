"""Dual Layer Aggregator (DLA) for Squeeze-MLLM (arXiv:2605.26111)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.squeeze_mllm.config import SqueezeMLLMConfig
from ltx_trainer.squeeze_mllm.lap import LayerwiseAttentionPooling


@dataclass(frozen=True)
class DLAOutput:
    text_agg: Tensor  # [B, L_text, C]
    image_agg: Tensor  # [B, L_img, C]


class DualLayerAggregator(nn.Module):
    """Separate LAP modules for text and image MLLM tokens (Sec. 3.3)."""

    def __init__(self, cfg: SqueezeMLLMConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SqueezeMLLMConfig()
        self.text_lap = LayerwiseAttentionPooling(cfg.embed_dim, cfg.num_attn_heads)
        self.image_lap = LayerwiseAttentionPooling(cfg.embed_dim, cfg.num_attn_heads)

    def forward(self, text_layers: Tensor, image_layers: Tensor) -> DLAOutput:
        """``text_layers`` / ``image_layers``: ``[B, M, L, C]`` per modality."""
        return DLAOutput(
            text_agg=self.text_lap(text_layers),
            image_agg=self.image_lap(image_layers),
        )


def synthetic_mllm_layer_stack(
    *,
    batch: int,
    num_layers: int,
    seq_len: int,
    channels: int,
    seed: int,
    modality_bias: float = 0.0,
) -> Tensor:
    """Toy layer features: early layers emphasize structure, late layers semantics."""
    torch.manual_seed(seed)
    base = torch.randn(batch, num_layers, seq_len, channels) * 0.05
    layer_idx = torch.linspace(0.0, 1.0, num_layers).view(1, num_layers, 1, 1)
    # Early vs late emphasis differs by modality.
    if modality_bias >= 0:
        base = base + modality_bias * (1.0 - layer_idx) * torch.randn(batch, 1, seq_len, channels)
    else:
        base = base + (-modality_bias) * layer_idx * torch.randn(batch, 1, seq_len, channels)
    return base
