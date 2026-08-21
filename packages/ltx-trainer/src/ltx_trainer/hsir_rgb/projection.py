"""Linear spectral projection + QR stability (Eqs. 2–5, arXiv:2605.24769)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


def encoder_decoder_from_unconstrained(e_tilde: Tensor) -> tuple[Tensor, Tensor]:
    """Map unconstrained ``\\tilde{E}`` ``[Kc, C]`` to orthonormal ``E = Q`` and ``F = Q^T`` (reduced QR)."""
    if e_tilde.dim() != 2:
        raise ValueError(f"e_tilde must be [Kc, C], got {tuple(e_tilde.shape)}")
    q, _ = torch.linalg.qr(e_tilde, mode="reduced")
    return q, q.mT


def encode_groups(y: Tensor, e: Tensor, inner_c: int) -> Tensor:
    """``y`` ``[B, C, H, W]``, ``E`` ``[Kc, C]`` → stacked latent groups ``[B, K, c, H, W]``."""
    if y.dim() != 4:
        raise ValueError(f"y must be [B,C,H,W], got {tuple(y.shape)}")
    b, C, h, w = y.shape
    kc, c_in = e.shape
    if c_in != C:
        raise ValueError(f"E second dim {c_in} must match C={C}")
    if kc % inner_c != 0:
        raise ValueError("Kc must be divisible by inner_c")
    flat = y.permute(0, 2, 3, 1).reshape(-1, C)
    z_flat = flat @ e.mT
    k = kc // inner_c
    z = z_flat.view(b, h, w, k, inner_c).permute(0, 3, 4, 1, 2).contiguous()
    return z


def decode_stacked(z_groups: Tensor, f_matrix: Tensor) -> Tensor:
    """``z_groups`` ``[B, K, c, H, W]``, ``F`` ``[C, Kc]`` → ``[B, C, H, W]``."""
    b, k, c, h, w = z_groups.shape
    flat = z_groups.permute(0, 3, 4, 1, 2).reshape(-1, k * c)
    x_flat = flat @ f_matrix.mT
    C = f_matrix.shape[0]
    return x_flat.view(b, h, w, C).permute(0, 3, 1, 2).contiguous()


def hyperspectral_denoise(
    y: Tensor,
    e: Tensor,
    f_matrix: Tensor,
    inner: Callable[[Tensor], Tensor],
    *,
    inner_c: int,
) -> Tensor:
    """Eq. (2): aggregate ``F_k D(E_k y)`` with shared inner denoiser ``D``."""
    z = encode_groups(y, e, inner_c)
    b, k, c, h, w = z.shape
    z_batched = z.reshape(b * k, c, h, w)
    z_den_b = inner(z_batched)
    z_den = z_den_b.reshape(b, k, c, h, w)
    return decode_stacked(z_den, f_matrix)


class SpectralPnPAdapter(nn.Module):
    """Trainable ``\\tilde{E}`` only; forward uses QR to obtain ``E, F`` (paper Sec. 2)."""

    def __init__(self, num_bands: int, num_groups: int, inner_channels: int) -> None:
        super().__init__()
        kc = num_groups * inner_channels
        if kc < num_bands:
            raise ValueError("Need Kc >= C for exact FE = I_C reconstruction under QR design.")
        self.num_bands = num_bands
        self.num_groups = num_groups
        self.inner_channels = inner_channels
        self.register_parameter("e_tilde", nn.Parameter(torch.randn(kc, num_bands) * 0.02))

    def encode_decode_matrices(self) -> tuple[Tensor, Tensor]:
        return encoder_decoder_from_unconstrained(self.e_tilde)

    def forward(self, y: Tensor, inner: Callable[[Tensor], Tensor] | None = None) -> Tensor:
        inner = inner or (lambda z: z)
        e, f = self.encode_decode_matrices()
        return hyperspectral_denoise(y, e, f, inner, inner_c=self.inner_channels)
