"""Dual-signal KV-cache compression (DCP) — Sec. 3.3, Eq. (1)–(4)."""

from __future__ import annotations

import math
from typing import Iterable


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo = min(xs)
    hi = max(xs)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def attention_importance(attn_h_by_p: list[list[float]]) -> list[float]:
    """I_att over tokens (Eq. 1 style).

    The paper aggregates last-layer attention over heads; exact tensor shapes vary by model.
    This stub assumes inputs are already collapsed to per-head token weights (H x P) and returns mean over H.
    """
    if not attn_h_by_p:
        return []
    P = len(attn_h_by_p[0])
    if P == 0:
        return []
    H = len(attn_h_by_p)
    out = [0.0 for _ in range(P)]
    for h in range(H):
        row = attn_h_by_p[h]
        for i in range(P):
            out[i] += float(row[i])
    return [x / H for x in out]


def dft_frequency_magnitudes(tokens_by_dim: list[list[float]]) -> list[list[float]]:
    """Tiny DFT magnitudes along token axis: returns |Z_fft| with shape (P x D) (Eq. 2 narrative).

    This is intentionally small and CPU-only for tests; not meant for large sequences.
    """
    P = len(tokens_by_dim)
    if P == 0:
        return []
    D = len(tokens_by_dim[0]) if tokens_by_dim[0] else 0
    if D == 0:
        return [[] for _ in range(P)]

    # For each frequency bin k (0..P-1), compute magnitude per dim d.
    out: list[list[float]] = []
    for k in range(P):
        mags = []
        for d in range(D):
            re = 0.0
            im = 0.0
            for n in range(P):
                angle = -2.0 * math.pi * k * n / P
                x = float(tokens_by_dim[n][d])
                re += x * math.cos(angle)
                im += x * math.sin(angle)
            mags.append((re * re + im * im) ** 0.5)
        out.append(mags)
    return out


def frequency_importance(keys_p_by_d: list[list[float]]) -> list[float]:
    """I_fft per token (Eq. 3 style).

    The paper applies FFT along each token dimension across {k_i}; this stub uses DFT magnitudes and mean-pools.
    """
    Z = dft_frequency_magnitudes(keys_p_by_d)
    if not Z:
        return []
    P = len(Z)
    D = len(Z[0]) if Z[0] else 0
    if D == 0:
        return [0.0 for _ in range(P)]
    return [sum(Z[i]) / D for i in range(P)]


def fuse_dual_signal(att_scores: list[float], fft_scores: list[float], *, alpha: float) -> list[float]:
    """I = α·norm(I_att) + (1-α)·norm(I_fft) (Eq. 4)."""
    if len(att_scores) != len(fft_scores):
        raise ValueError("att_scores and fft_scores must have same length")
    a = _minmax([float(x) for x in att_scores])
    f = _minmax([float(x) for x in fft_scores])
    out = []
    for i in range(len(a)):
        out.append(alpha * a[i] + (1.0 - alpha) * f[i])
    return out


def topk_indices(scores: list[float], k: int) -> list[int]:
    if k <= 0 or not scores:
        return []
    k = min(k, len(scores))
    # Stable sort by score desc, then index asc.
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def retain_ratio_to_k(num_tokens: int, rho: float) -> int:
    return max(0, min(num_tokens, int(math.floor(float(rho) * float(num_tokens)))))


def compress_indices(scores: list[float], *, rho: float) -> list[int]:
    """Return kept token indices after pruning by ratio rho."""
    k = retain_ratio_to_k(len(scores), rho)
    return topk_indices(scores, k)


def compress_kv_stub(
    *,
    keys_p_by_d: list[list[float]],
    attn_h_by_p: list[list[float]],
    alpha: float,
    rho: float,
) -> dict[str, object]:
    """End-to-end DCP stub returning indices and fused scores."""
    att = attention_importance(attn_h_by_p)
    fft = frequency_importance(keys_p_by_d)
    fused = fuse_dual_signal(att, fft, alpha=alpha)
    keep = compress_indices(fused, rho=rho)
    return {"keep_indices": keep, "fused_scores": fused, "att_scores": att, "fft_scores": fft}


def mean_pool(vectors: Iterable[list[float]]) -> list[float]:
    vs = list(vectors)
    if not vs:
        return []
    D = len(vs[0])
    out = [0.0 for _ in range(D)]
    for v in vs:
        for d in range(D):
            out[d] += float(v[d])
    n = float(len(vs))
    return [x / n for x in out]

