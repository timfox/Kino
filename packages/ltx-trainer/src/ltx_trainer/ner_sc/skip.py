"""Embedding-level frame skip (Eq. 7)."""

from __future__ import annotations

import numpy as np


def normalized_embedding_l1(z_i: np.ndarray, z_k: np.ndarray) -> float:
    """``(1/|z|) ||z_i - z_k||_1`` for compact latent embeddings."""
    a = np.asarray(z_i, dtype=np.float64).ravel()
    b = np.asarray(z_k, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("embeddings must have same shape")
    denom = max(1, a.size)
    return float(np.abs(a - b).sum() / denom)


def should_skip_decode(
    z_i: np.ndarray,
    z_k: np.ndarray,
    *,
    tau: float = 0.005,
) -> bool:
    """Return True when decoder can be bypassed (Eq. 7)."""
    return normalized_embedding_l1(z_i, z_k) < tau


def decode_with_skip(
    embeddings: list[np.ndarray],
    decode_fn,
    *,
    tau: float = 0.005,
) -> tuple[list[np.ndarray | None], dict[str, float]]:
    """Run embedding-level skip across a frame sequence.

    ``decode_fn(z)`` returns a reconstruction array; skipped frames reuse last decode.
    """
    outputs: list[np.ndarray | None] = []
    z_ref: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    n_skip = 0
    for z in embeddings:
        if z_ref is not None and should_skip_decode(z, z_ref, tau=tau):
            outputs.append(last_frame)
            n_skip += 1
        else:
            frame = decode_fn(z)
            outputs.append(frame)
            z_ref = z.copy()
            last_frame = frame
    n = len(embeddings)
    return outputs, {
        "skip_rate": n_skip / n if n else 0.0,
        "frames_decoded": n - n_skip,
        "frames_skipped": n_skip,
        "tau": tau,
    }
