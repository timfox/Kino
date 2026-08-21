"""RI + magnitude combined loss (Eq. 43–45)."""

from __future__ import annotations

import numpy as np


def ri_loss(pred_r: np.ndarray, pred_i: np.ndarray, tgt_r: np.ndarray, tgt_i: np.ndarray) -> float:
    return float(np.linalg.norm(pred_r - tgt_r) ** 2 + np.linalg.norm(pred_i - tgt_i) ** 2)


def mag_loss(
    pred_r: np.ndarray,
    pred_i: np.ndarray,
    tgt_r: np.ndarray,
    tgt_i: np.ndarray,
) -> float:
    pred_mag = np.sqrt(pred_r**2 + pred_i**2 + 1e-8)
    tgt_mag = np.sqrt(tgt_r**2 + tgt_i**2 + 1e-8)
    return float(np.linalg.norm(pred_mag - tgt_mag))


def combined_loss(
    pred_r: np.ndarray,
    pred_i: np.ndarray,
    tgt_r: np.ndarray,
    tgt_i: np.ndarray,
    *,
    beta: float = 0.5,
) -> dict[str, float]:
    l_ri = ri_loss(pred_r, pred_i, tgt_r, tgt_i)
    l_mag = mag_loss(pred_r, pred_i, tgt_r, tgt_i)
    total = beta * l_ri + (1.0 - beta) * l_mag
    return {"l_ri": l_ri, "l_mag": l_mag, "total": total}
