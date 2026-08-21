"""Reference LoRA self-attention stub (paper Eq. 4–7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceLoRAConfig:
    d_model: int = 64
    lora_rank: int = 8
    scale: float = 1.0


def lora_project(x: np.ndarray, w: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Y = (W + B @ A) X with low-rank BA."""
    base = x @ w.T
    delta = (x @ a.T) @ b.T
    return base + delta


def reference_self_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    q_ref: np.ndarray,
    k_ref: np.ndarray,
    v_ref: np.ndarray,
    *,
    d_head: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Zr and Z with concatenated reference keys/values (Eq. 6)."""
    d = d_head or max(int(np.sqrt(q.shape[-1])), 1)
    scale = 1.0 / np.sqrt(d)

    z_ref = _softmax(q_ref @ k_ref.T * scale) @ v_ref
    k_cat = np.concatenate([k, k_ref], axis=0)
    v_cat = np.concatenate([v, v_ref], axis=0)
    z = _softmax(q @ k_cat.T * scale) @ v_cat
    return z, z_ref


def inject_embedding(z: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """Add projected face/timbre embedding (Eq. 7)."""
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    if embedding.shape[-1] != z.shape[-1]:
        # simple linear pad/truncate for stub
        out = np.zeros(z.shape[-1], dtype=np.float64)
        n = min(out.size, embedding.size)
        out[:n] = embedding.reshape(-1)[:n]
        embedding = out.reshape(1, -1)
    return z + embedding.mean(axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)
