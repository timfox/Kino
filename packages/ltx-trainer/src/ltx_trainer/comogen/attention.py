"""Motion Layer identification via attention scores (Sec. 4.1, Eq. 2)."""

from __future__ import annotations

import torch
from torch import Tensor


def attention_score(
    mask: Tensor,
    attn: Tensor,
) -> float:
    """S_l = sum(M ⊗ A) / sum(A) (Eq. 2).

    mask: [F, H, W] binary subject region
    attn: [F, H, W] normalized attention map for subject token
    """
    m = mask.to(dtype=attn.dtype)
    num = (m * attn).sum()
    den = attn.sum().clamp(min=1e-8)
    return float((num / den).item())


def layer_attention_scores(
    masks: Tensor,
    attentions: Tensor,
) -> list[float]:
    """Compute S_l for each layer.

    masks: [F, H, W]
    attentions: [L, F, H, W]
    """
    scores: list[float] = []
    for layer in range(attentions.shape[0]):
        scores.append(attention_score(masks, attentions[layer]))
    return scores


def identify_motion_layers(
    scores: list[float],
    num_motion_layers: int = 11,
) -> tuple[list[int], list[int]]:
    """Return indices of top-k Motion Layers and remaining Non-Motion Layers."""
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    motion = sorted(ranked[:num_motion_layers])
    non_motion = sorted(ranked[num_motion_layers:])
    return motion, non_motion


def demo_attention_maps(
    frames: int = 8,
    height: int = 16,
    width: int = 16,
    num_layers: int = 20,
) -> tuple[Tensor, Tensor]:
    """Synthetic mask + attention where later layers align better (Fig. 3 pattern)."""
    mask = torch.zeros(frames, height, width)
    mask[:, 4:12, 4:12] = 1.0
    attn = torch.rand(num_layers, frames, height, width) * 0.1
    for layer in range(num_layers):
        boost = (layer / max(num_layers - 1, 1)) ** 2
        attn[layer] = attn[layer] + boost * mask
    attn = attn / (attn.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return mask, attn
