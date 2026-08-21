"""Differentiable JPEG rate proxy (§2, sparsity-aware RLE surrogate)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def block_dct_proxy(coeffs: Tensor, q: Tensor) -> Tensor:
    """Smooth bpp proxy per 8×8 block; coeffs (B, C, 8, 8), q (C, 8, 8)."""
    scaled = coeffs.abs() / q.clamp(min=1.0)
    gate = torch.tanh(coeffs.pow(2))
    per_coeff = gate * torch.log2(1.0 + scaled)
    huffman_overhead = 0.25  # fixed per-block constant (paper)
    return per_coeff.sum(dim=(-2, -1)) + huffman_overhead


def rate_proxy_bpp(
    latent: Tensor,
    q: Tensor,
    *,
    alpha_calib: float = 1.0,
    pixels: int | None = None,
) -> Tensor:
    """
    Calibrated bits-per-pixel proxy for training Eq. (5).

    ``latent`` is treated as pseudo-DCT coefficients for smoke tests.
    """
    b, c, h, w = latent.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return torch.tensor(0.01 * alpha_calib, device=latent.device, dtype=latent.dtype)
    latent = latent[..., :h8, :w8]
    blocks = latent.unfold(2, 8, 8).unfold(3, 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    blocks = blocks.view(-1, c, 8, 8)
    bits = block_dct_proxy(blocks, q).sum()
    denom = pixels if pixels is not None else (b * h * w)
    return alpha_calib * bits / max(denom, 1)
