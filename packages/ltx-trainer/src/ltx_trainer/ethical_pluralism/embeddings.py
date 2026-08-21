"""Triple-BERT supervector stub (1920D) — Sec. semantic stream (arXiv:2605.28707)."""

from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np

# Paper dims: MiniLM 384 + distilRoBERTa 768 + mpnet 768 = 1920
_STREAM_DIMS_FULL = (384, 768, 768)
_STREAM_DIMS_STUB = (64, 64, 64)  # fast deterministic stub (layout matches paper structure)

_EMBED_CACHE: dict[str, np.ndarray] = {}


def _stream_hash(text: str, dim: int, salt: str) -> np.ndarray:
    """Deterministic pseudo-embedding from text (no transformer download)."""
    out = np.zeros(dim, dtype=np.float64)
    payload = f"{salt}:{text}".encode("utf-8")
    for i in range(dim):
        h = hashlib.sha256(payload + struct.pack("<I", i)).digest()
        out[i] = (int.from_bytes(h[:4], "little") / 2**32) * 2.0 - 1.0
    n = np.linalg.norm(out)
    if n > 1e-9:
        out /= n
    return out


def triple_bert_supervector(
    raw_text: str,
    summary: str = "",
    *,
    cfg_names: Sequence[str] | None = None,
    stub_fast: bool = True,
) -> np.ndarray:
    """
    Concatenate three sentence-transformer streams (stub).

    Production would call all-MiniLM-L6-v2, distilRoBERTa, and mpnet on
    raw + summary inputs per the paper.
    """
    _ = cfg_names
    key = f"{raw_text}\0{summary}\0{stub_fast}"
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    dims = _STREAM_DIMS_STUB if stub_fast else _STREAM_DIMS_FULL
    parts = [
        _stream_hash(raw_text, dims[0], "minilm"),
        _stream_hash(summary or raw_text[:512], dims[1], "distilroberta"),
        _stream_hash(f"{raw_text}\n{summary}", dims[2], "mpnet"),
    ]
    vec = np.concatenate(parts)
    _EMBED_CACHE[key] = vec
    return vec


def project_supervector(vec: np.ndarray, dim: int, seed: int = 42) -> np.ndarray:
    """Fixed random projection for lightweight ensemble training."""
    rng = np.random.default_rng(seed)
    if vec.shape[0] <= dim:
        out = np.zeros(dim)
        out[: vec.shape[0]] = vec
        return out
    proj = rng.standard_normal((dim, vec.shape[0])) / np.sqrt(vec.shape[0])
    return proj @ vec
