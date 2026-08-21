"""PMV and base avatar losses (Eq. 10)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ltx_trainer.prism_avatar.config import PrismAvatarConfig


def l1_masked(pred: NDArray[np.floating], target: NDArray[np.floating], mask: NDArray[np.floating]) -> float:
    m = mask > 0.5
    if not np.any(m):
        return 0.0
    return float(np.mean(np.abs(pred[m] - target[m])))


def alpha_translucency_smear(alpha: NDArray[np.floating], support: NDArray[np.floating]) -> float:
    """L_s via 4 α (1−α) on support."""
    resp = 4.0 * alpha * (1.0 - alpha)
    s = support > 0.5
    if not np.any(s):
        return 0.0
    return float(np.mean(resp[s]))


def pmv_loss_bundle(
    pred_rgb: NDArray[np.floating],
    pred_alpha: NDArray[np.floating],
    target_rgb: NDArray[np.floating],
    target_alpha: NDArray[np.floating],
    mesh_mask: NDArray[np.floating],
    safe_mask: NDArray[np.floating],
    cfg: PrismAvatarConfig,
    *,
    stage: str = "side",
) -> dict[str, float]:
    """L_pmv components with paper weights."""
    w = cfg.pmv_weights_side if stage == "side" else cfg.pmv_weights_color
    lam_i, lam_a, lam_bg, lam_mesh, lam_e, lam_s = w
    li = l1_masked(pred_rgb, target_rgb, safe_mask)
    la = l1_masked(pred_alpha, target_alpha, safe_mask)
    outside = (1.0 - mesh_mask) * (pred_alpha > 0.03)
    lbg = float(np.mean(outside)) if np.any(outside) else 0.0
    lmesh = float(np.mean(pred_alpha * (1.0 - mesh_mask)))
    edge_band = dilate_edge(safe_mask)
    le = l1_masked(pred_alpha, target_alpha, edge_band)
    ls = alpha_translucency_smear(pred_alpha, mesh_mask)
    total = lam_i * li + lam_a * la + lam_bg * lbg + lam_mesh * lmesh + lam_e * le + lam_s * ls
    return {
        "L_pmv": total,
        "L_I": li,
        "L_alpha": la,
        "L_bg": lbg,
        "L_mesh": lmesh,
        "L_edge": le,
        "L_smear": ls,
    }


def pmv_gamma(iteration: int, cfg: PrismAvatarConfig) -> float:
    """Scheduled γ(k) ramp for PMV term."""
    if iteration < cfg.pmv_start_iter:
        return 0.0
    t = min(1.0, (iteration - cfg.pmv_start_iter) / max(cfg.pmv_ramp_iters, 1))
    return cfg.pmv_gamma_cap * t


def dilate_edge(mask: NDArray[np.floating], band: int = 3) -> NDArray[np.floating]:
    h, w = mask.shape[:2]
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            y0, y1 = max(0, y - band), min(h, y + band + 1)
            x0, x1 = max(0, x - band), min(w, x + band + 1)
            if np.any(mask[y0:y1, x0:x1] > 0.5):
                out[y, x] = 1.0
    return out
