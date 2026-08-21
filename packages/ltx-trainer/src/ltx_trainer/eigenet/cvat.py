"""Cross-view Alternate-attention Transformer (CVAT) stubs (Sec. III-E)."""

from __future__ import annotations

from typing import Any

import numpy as np


def stack_view_tokens(
    geometric: np.ndarray,
    acoustic: np.ndarray,
) -> np.ndarray:
    """Vi = Concat(Gi, Ai) along sequence dim — Eq. (3)."""
    g = np.asarray(geometric)
    a = np.asarray(acoustic)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if g.shape[1] != a.shape[1]:
        width = max(g.shape[1], a.shape[1])
        if g.shape[1] < width:
            g = np.pad(g, ((0, 0), (0, width - g.shape[1])))
        if a.shape[1] < width:
            a = np.pad(a, ((0, 0), (0, width - a.shape[1])))
    return np.concatenate([g, a], axis=0)


def local_attention(views: list[np.ndarray]) -> list[np.ndarray]:
    """Eq. (5): MSAlocal within each view."""
    out: list[np.ndarray] = []
    for v in views:
        # Identity residual stub — production uses multi-head self-attention
        out.append(v + v * 0.0)
    return out


def global_attention(sequence: np.ndarray) -> np.ndarray:
    """Eq. (7): MSAglobal over all views."""
    return sequence  # stub residual path in full model


def alternate_attention_block(views: list[np.ndarray]) -> list[np.ndarray]:
    """One block: local then global (Sec. III-E.5)."""
    local_out = local_attention(views)
    stacked = np.concatenate(local_out, axis=0)
    global_out = global_attention(stacked)
    # Repartition equally (stub)
    sizes = [v.shape[0] for v in views]
    splits = np.split(global_out, np.cumsum(sizes)[:-1], axis=0)
    return [s + g for s, g in zip(splits, local_out, strict=True)]


def cvat_forward(
    h0: np.ndarray,
    n_views: int,
    n_blocks: int = 6,
) -> np.ndarray:
    """H(0) from N+1 views; apply alternating blocks — Eq. (4)-(7)."""
    per_view_len = h0.shape[0] // max(n_views, 1)
    views = [h0[i * per_view_len : (i + 1) * per_view_len] for i in range(n_views)]
    for _ in range(n_blocks):
        views = alternate_attention_block(views)
    return np.concatenate(views, axis=0)


def attention_variant_card() -> dict[str, str]:
    return {
        "alternate": "Local intra-view then global cross-view; best on AR/HAA benchmarks.",
        "self": "Flatten all tokens; soft interpolator over reference acoustics (Tab. IV masking).",
        "cross": "Target queries reference K/V; weakest on broadband metrics.",
    }
