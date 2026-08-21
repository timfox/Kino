"""Length-preserving skip/repeat latent augmentations (Eq. 4–5, arXiv:2605.22083)."""

from __future__ import annotations

import torch
from torch import Tensor


def repeat_overwrite(
    x: Tensor,
    *,
    src_start: int,
    tgt_start: int,
    length: int,
) -> Tensor:
    """Repeat augmentation: overwrite target region with a source span (Eq. 4).

    x_rep[k:k+ℓ] ← x[s:s+ℓ], s != k. Sequence length stays fixed.
    """
    if x.dim() < 2:
        raise ValueError("expected x with shape (..., T)")
    t = int(x.shape[-1])
    if length <= 0:
        raise ValueError("length must be positive")
    if not (0 <= src_start < t and 0 <= tgt_start < t):
        raise ValueError("span starts must be in [0, T)")
    if src_start + length > t or tgt_start + length > t:
        raise ValueError("span exceeds sequence length")
    if src_start == tgt_start:
        raise ValueError("src_start must differ from tgt_start")
    out = x.clone()
    out[..., tgt_start : tgt_start + length] = x[..., src_start : src_start + length]
    return out


def skip_shift_silence(
    x: Tensor,
    *,
    start: int,
    length: int,
    silence_latent: Tensor | None = None,
) -> Tensor:
    """Skip augmentation: shift future context forward and pad tail with silence (Eq. 5).

    x_skip[s1:T-ℓ] ← x[s1+ℓ:T], and x_skip[T-ℓ:T] ← x_sil.
    """
    if x.dim() < 2:
        raise ValueError("expected x with shape (..., T)")
    t = int(x.shape[-1])
    if length <= 0 or length >= t:
        raise ValueError("length must be in [1, T-1]")
    if not (0 <= start < t):
        raise ValueError("start must be in [0, T)")
    if start + length > t:
        raise ValueError("skip span exceeds sequence length")

    out = x.clone()
    out[..., start : t - length] = x[..., start + length : t]

    if silence_latent is None:
        out[..., t - length : t] = 0
        return out

    sil = silence_latent
    if sil.dim() < 1:
        raise ValueError("silence_latent must have at least one dimension")
    if sil.shape[-1] not in (1, length):
        raise ValueError("silence_latent last dim must be 1 or length")
    # Broadcast silence to match tail slice.
    if sil.shape[-1] == 1:
        sil = sil.expand(*sil.shape[:-1], length)
    out[..., t - length : t] = sil
    return out


def augment_batch(
    x: Tensor,
    *,
    p_repeat: float = 0.5,
    silence_latent: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, int]]:
    """Stochastic in-batch augmentation: pick repeat or skip per sample.

    This is a simplified stub: spans are sampled uniformly over valid indices and
    lengths use a small discrete set to keep tests deterministic.
    """
    if x.dim() != 3:
        raise ValueError("expected x with shape (B, C, T)")
    b, _c, t = x.shape
    if t < 8:
        raise ValueError("T too small for augmentation")
    if not (0.0 <= p_repeat <= 1.0):
        raise ValueError("p_repeat must be in [0, 1]")

    rng = generator
    out = x.clone()
    counts = {"repeat": 0, "skip": 0}

    # Small, fixed length candidates for a stable stub.
    lengths = torch.tensor([max(2, t // 20), max(3, t // 12), max(4, t // 8)], device=x.device)

    for i in range(b):
        u = float(torch.rand((), generator=rng, device=x.device).item())
        l = int(lengths[int(torch.randint(0, len(lengths), (), generator=rng, device=x.device).item())].item())
        l = min(l, t - 1)

        if u < p_repeat:
            # sample src/tgt ensuring different
            src = int(torch.randint(0, t - l, (), generator=rng, device=x.device).item())
            tgt = int(torch.randint(0, t - l, (), generator=rng, device=x.device).item())
            for _ in range(32):
                if tgt != src:
                    break
                tgt = int(torch.randint(0, t - l, (), generator=rng, device=x.device).item())
            if tgt == src:
                tgt = (src + max(1, l)) % max(1, t - l)
            out[i] = repeat_overwrite(out[i], src_start=src, tgt_start=tgt, length=l)
            counts["repeat"] += 1
        else:
            start = int(torch.randint(0, t - l, (), generator=rng, device=x.device).item())
            out[i] = skip_shift_silence(out[i], start=start, length=l, silence_latent=silence_latent)
            counts["skip"] += 1

    return out, counts

