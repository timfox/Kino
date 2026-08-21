"""HoliTok downstream AR+DiT unified generation-understanding proxies."""

from __future__ import annotations

import numpy as np


def patchify_latents(latents: np.ndarray, patch_size: int = 4) -> np.ndarray:
    """Group consecutive frames into patch tokens (Table 4 patch size 4)."""
    z = np.asarray(latents, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    t, d = z.shape
    usable = (t // patch_size) * patch_size
    if usable == 0:
        return z[:1].reshape(1, -1)
    blocks = z[:usable].reshape(-1, patch_size * d)
    return blocks


def flow_matching_loss(
    velocity_pred: np.ndarray,
    velocity_target: np.ndarray,
) -> float:
    """Eq. 11 — conditional flow-matching MSE."""
    vp = np.asarray(velocity_pred, dtype=np.float64)
    vt = np.asarray(velocity_target, dtype=np.float64)
    return float(np.mean((vp - vt) ** 2))


def understanding_ce_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Eq. 9 — autoregressive cross-entropy proxy (softmax over vocab dim)."""
    lg = np.asarray(logits, dtype=np.float64)
    tg = np.asarray(targets, dtype=np.int64).ravel()
    if lg.ndim == 1:
        lg = lg.reshape(1, -1)
    losses: list[float] = []
    for i, tok in enumerate(tg):
        row = lg[i % lg.shape[0]]
        ex = np.exp(row - np.max(row))
        prob = ex / (ex.sum() + 1e-8)
        idx = int(tok) % prob.size
        losses.append(-float(np.log(prob[idx] + 1e-8)))
    return float(np.mean(losses)) if losses else 0.0


def generation_objective(
    velocity_pred: np.ndarray,
    velocity_target: np.ndarray,
    eos_logits: np.ndarray,
    eos_targets: np.ndarray,
    *,
    lambda_eos: float = 1.0,
) -> dict[str, float]:
    """Eq. 12 — L_FM + λ_eos L_eos."""
    l_fm = flow_matching_loss(velocity_pred, velocity_target)
    eos_p = 1.0 / (1.0 + np.exp(-np.asarray(eos_logits, dtype=np.float64)))
    eos_t = np.asarray(eos_targets, dtype=np.float64)
    l_eos = float(np.mean(-(eos_t * np.log(eos_p + 1e-8) + (1 - eos_t) * np.log(1 - eos_p + 1e-8))))
    return {"l_fm": l_fm, "l_eos": l_eos, "total": l_fm + lambda_eos * l_eos}
