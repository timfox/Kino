"""P2PSVQ plain + pseudo VQ stubs (arXiv:2606.05876)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ltx_trainer.p2psyncodec.config import P2PSynCodecConfig


def plain_vq_token(encoded: np.ndarray, codebook: np.ndarray) -> int:
    """Eq. (1): d_pl = argmin_m ||e - w_m||^2."""
    distances = np.sum((codebook - encoded.reshape(1, -1)) ** 2, axis=1)
    return int(np.argmin(distances))


def bitrate_bps(*, fs: int, cfg: P2PSynCodecConfig | None = None) -> float:
    """Eq. (2): only plain VQ contributes — fs/D * log2(M_pl)."""
    c = cfg or P2PSynCodecConfig()
    return (fs / c.downsampling_rate) * math.log2(c.plain_codebook_size)


def bitrate_kbps(*, fs: int, cfg: P2PSynCodecConfig | None = None) -> float:
    return bitrate_bps(fs=fs, cfg=cfg) / 1000.0


def codebook_lookup(codebook: np.ndarray, token: int) -> np.ndarray:
    return codebook[token]


def pseudo_vq_predict_logits(
    *,
    plain_embedding: np.ndarray,
    prior_embeddings: list[np.ndarray],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Eq. (3)–(4) stub: return logits for argmax auxiliary token."""
    gen = rng or np.random.default_rng(0)
    context = plain_embedding + sum(prior_embeddings)
    return gen.normal(context.mean(), 0.1, size=128)


def pseudo_vq_token(logits: np.ndarray) -> int:
    """Eq. (4): d_ps = argmax_i z_i."""
    return int(np.argmax(logits))


def synergistic_quantized_vector(
    plain_embedding: np.ndarray,
    pseudo_embeddings: list[np.ndarray],
) -> np.ndarray:
    """Eq. (5): e_hat = L(W_pl, d_pl) + sum_n L(W_ps, d_ps)."""
    out = plain_embedding.copy()
    for emb in pseudo_embeddings:
        out = out + emb
    return out


def pseudo_vq_ce_loss(predicted_probs: np.ndarray, target_one_hot: np.ndarray) -> float:
    """Eq. (7): cross-entropy between softmax logits and teacher one-hot."""
    eps = 1e-9
    p = np.clip(predicted_probs, eps, 1.0)
    return float(-np.sum(target_one_hot * np.log(p)))


def teacher_forcing_train_step(
    teacher_tokens: list[int],
    codebooks: list[np.ndarray],
    *,
    n_pseudo: int,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Eq. (6)–(7): independent CE training for each pseudo VQ."""
    gen = rng or np.random.default_rng(0)
    losses: list[float] = []
    for n in range(1, n_pseudo + 1):
        context = sum(codebooks[i][teacher_tokens[i]] for i in range(n))
        logits = gen.normal(0, 1, codebooks[n].shape[0])
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        target = np.zeros_like(probs)
        target[teacher_tokens[n]] = 1.0
        losses.append(pseudo_vq_ce_loss(probs, target))
    return {"pseudo_vq_losses": losses, "mean_ce": float(np.mean(losses))}
