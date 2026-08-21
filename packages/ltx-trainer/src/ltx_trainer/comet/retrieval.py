"""In-domain retrieval R@K from PLSHead embeddings (COMET Table III)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.pls import pls_svd
from ltx_trainer.comet.plshead import plshead_truncate


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def recall_at_k(query: np.ndarray, gallery: np.ndarray, *, k: int, gt_index: int) -> float:
    q = query / (np.linalg.norm(query) + 1e-8)
    g = _normalize_rows(gallery)
    sims = g @ q
    top = np.argsort(-sims)[:k]
    return float(gt_index in top)


def text_to_audio_retrieval(
    text: np.ndarray,
    audio: np.ndarray,
    *,
    head_size: int,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Text→audio retrieval on paired matrices (diagonal positives)."""
    decomp = pls_svd(text, audio)
    n = text.shape[0]
    recalls = {f"R{k}": 0.0 for k in ks}
    for i in range(n):
        t100 = plshead_truncate(text[i], mean=decomp["t_mean"], directions=decomp["U"], head_size=head_size)
        gallery = np.stack(
            [
                plshead_truncate(audio[j], mean=decomp["a_mean"], directions=decomp["V"], head_size=head_size)
                for j in range(n)
            ],
            axis=0,
        )
        for k in ks:
            recalls[f"R{k}"] += recall_at_k(t100, gallery, k=k, gt_index=i)
    return {k: v / max(n, 1) for k, v in recalls.items()}


def retrieval_smoke(cfg: CometConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or CometConfig()
    rng = np.random.default_rng(seed)
    n, c = 32, cfg.embed_dim
    shared = rng.standard_normal((n, cfg.head_size))
    u = rng.standard_normal((c, cfg.head_size))
    text = shared @ u.T + 0.05 * rng.standard_normal((n, c))
    audio = shared @ u.T + 0.08 * rng.standard_normal((n, c)) - 0.1
    r = text_to_audio_retrieval(text, audio, head_size=cfg.head_size)
    return {"n_pairs": n, "head_size": cfg.head_size, **{k: round(v, 3) for k, v in r.items()}}
