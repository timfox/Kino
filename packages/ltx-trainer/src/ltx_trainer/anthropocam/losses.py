"""Content, style (Gram), and total-variation losses (Eqs. 2–5)."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from ltx_trainer.anthropocam.gram import gram_matrix


def content_loss(content_feats: Tensor, generated_feats: Tensor) -> Tensor:
    """Eq. 2 — MSE between feature maps at content layer."""
    return 0.5 * torch.mean((generated_feats - content_feats) ** 2)


def style_loss(
    style_grams: Mapping[str, Tensor],
    generated_feats: Mapping[str, Tensor],
    layer_weights: Mapping[str, float],
) -> Tensor:
    """Eq. 3 — weighted Gram matrix MSE across style layers."""
    total = generated_feats[list(generated_feats.keys())[0]].new_zeros(())
    for layer, w in layer_weights.items():
        if layer not in style_grams or layer not in generated_feats:
            continue
        g_x = gram_matrix(generated_feats[layer])
        g_a = style_grams[layer]
        n_l = g_x.shape[1]
        m_l = g_x.shape[2]
        layer_loss = torch.mean((g_x - g_a) ** 2) / (4.0 * (n_l**2) * (m_l**2))
        total = total + float(w) * layer_loss
    return total


def total_variation_loss(image: Tensor) -> Tensor:
    """Eq. 4 — anisotropic TV on generated RGB image."""
    dh = torch.mean(torch.abs(image[..., 1:, :] - image[..., :-1, :]))
    dw = torch.mean(torch.abs(image[..., :, 1:] - image[..., :, :-1]))
    return dh + dw


def total_loss(
    content: Tensor,
    style: Tensor,
    tv: Tensor,
    *,
    content_weight: float,
    style_weight: float,
    tv_weight: float,
) -> Tensor:
    """Eq. 5."""
    return content_weight * content + style_weight * style + tv_weight * tv
