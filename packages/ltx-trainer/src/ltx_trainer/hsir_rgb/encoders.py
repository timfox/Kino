"""Ablation encoders from Table 3 (arXiv:2605.24769)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from torch import Tensor

from ltx_trainer.hsir_rgb.projection import hyperspectral_denoise


class EncoderMode(str, Enum):
    SEQUENTIAL_MONO = "seq_mono"
    SEQUENTIAL_RGB = "seq_rgb"
    RANDOM = "random"
    PCA = "pca"
    GROUP_K1 = "group_k1"
    PROPOSED_QR = "proposed_qr"


def denoise_sequential(
    y: Tensor,
    inner: Callable[[Tensor], Tensor],
    *,
    inner_c: int,
) -> Tensor:
    """E = I_C style: no spectral mixing; denoise band groups of width ``inner_c`` in order."""
    if y.dim() != 4:
        raise ValueError("y must be [B,C,H,W]")
    b, C, h, w = y.shape
    out = torch.zeros_like(y)
    for start in range(0, C, inner_c):
        end = min(start + inner_c, C)
        width = end - start
        chunk = y[:, start:end]
        if width < inner_c:
            pad = torch.zeros(b, inner_c - width, h, w, device=y.device, dtype=y.dtype)
            chunk = torch.cat([chunk, pad], dim=1)
        den = inner(chunk)
        out[:, start:end] = den[:, :width]
    return out


def encoder_matrix_random(
    num_bands: int,
    num_groups: int,
    inner_c: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Permuted-identity rows → random band groupings (Table 3 ``Random``)."""
    kc = num_groups * inner_c
    perm = torch.randperm(num_bands, generator=generator)
    e = torch.zeros(kc, num_bands)
    for i in range(kc):
        e[i, perm[i % num_bands]] = 1.0
    return e


def encoder_matrix_pca(
    cube: Tensor,
    num_groups: int,
    inner_c: int,
) -> Tensor:
    """PCA spectral projection (Table 3 ``PCA proj.``); ``cube`` ``[B,C,H,W]`` or ``[C,H,W]``."""
    if cube.dim() == 3:
        cube = cube.unsqueeze(0)
    _, C, _, _ = cube.shape
    flat = cube.permute(0, 2, 3, 1).reshape(-1, C)
    kc = num_groups * inner_c
    q = min(kc, C)
    _, _, v = torch.pca_lowrank(flat, q=q, center=True)
    e = torch.zeros(kc, C, device=cube.device, dtype=cube.dtype)
    e[:q, :] = v.mT
    for i in range(q, kc):
        e[i, (i - q) % C] = 1.0
    return e


def encoder_matrix_group_k1(num_bands: int, inner_c: int = 3) -> tuple[Tensor, Tensor]:
    """Single RGB group (K = 1): ``E`` ``[c, C]`` selects first ``c`` bands, ``F`` pseudo-inverse."""
    e = torch.zeros(inner_c, num_bands)
    e[:, :inner_c] = torch.eye(inner_c)
    f = e.mT @ torch.linalg.inv(e @ e.mT + 1e-6 * torch.eye(inner_c))
    return e, f


def encoder_decoder_proposed(
    e_tilde: Tensor,
) -> tuple[Tensor, Tensor]:
    from ltx_trainer.hsir_rgb.projection import encoder_decoder_from_unconstrained

    return encoder_decoder_from_unconstrained(e_tilde)


def denoise_with_encoder(
    y: Tensor,
    e: Tensor,
    f_matrix: Tensor,
    inner: Callable[[Tensor], Tensor],
    *,
    inner_c: int,
) -> Tensor:
    return hyperspectral_denoise(y, e, f_matrix, inner, inner_c=inner_c)


def denoise_by_mode(
    y: Tensor,
    mode: EncoderMode,
    inner: Callable[[Tensor], Tensor],
    *,
    inner_c: int = 3,
    num_groups: int = 11,
    e_tilde: Tensor | None = None,
    pca_reference: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Route Table 3 encoder designs through one API."""
    C = y.shape[1]
    if mode == EncoderMode.SEQUENTIAL_MONO:
        return denoise_sequential(y, inner, inner_c=1)
    if mode == EncoderMode.SEQUENTIAL_RGB:
        return denoise_sequential(y, inner, inner_c=inner_c)
    if mode == EncoderMode.RANDOM:
        e = encoder_matrix_random(C, num_groups, inner_c, generator=generator)
        f = torch.linalg.pinv(e)
        return denoise_with_encoder(y, e, f, inner, inner_c=inner_c)
    if mode == EncoderMode.PCA:
        ref = pca_reference if pca_reference is not None else y
        e = encoder_matrix_pca(ref, num_groups, inner_c)
        f = torch.linalg.pinv(e)
        return denoise_with_encoder(y, e, f, inner, inner_c=inner_c)
    if mode == EncoderMode.GROUP_K1:
        e, f = encoder_matrix_group_k1(C, inner_c)
        return denoise_with_encoder(y, e, f, inner, inner_c=inner_c)
    if mode == EncoderMode.PROPOSED_QR:
        if e_tilde is None:
            raise ValueError("proposed_qr requires e_tilde")
        e, f = encoder_decoder_proposed(e_tilde)
        return denoise_with_encoder(y, e, f, inner, inner_c=inner_c)
    raise ValueError(f"Unknown mode {mode}")
