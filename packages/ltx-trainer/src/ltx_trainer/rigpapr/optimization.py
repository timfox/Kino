"""Two-phase video-guided optimization (Sec. 3.4, App. C.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.rigpapr.lbs import lbs_deform, rodrigues, se3_from_rotation, softmax_weights

# Table 5 — loss weights
PHASE1_WEIGHTS: dict[str, float] = {
    "Lrgb": 1.0,
    "LPIPS": 0.01,
    "L_delta_depth": 1.0,
    "Lrot": 0.1,
}

PHASE2_WEIGHTS: dict[str, float] = {
    "Lrgb": 1.0,
    "LPIPS": 0.01,
    "L_delta_depth_decay": 1.0,
    "Ltrack_decay": 0.005,
    "LARAP": 500.0,
    "Lcomposed_rot": 0.01,
    "Lcorr_l2": 5e-4,
    "Lcorr_smooth": 0.1,
    "Lweight_smooth": 0.03,
    "Lweight_sparse": 0.01,
}


def descendant_weights(num_bones: int, *, seed: int = 0) -> np.ndarray:
    """Toy descendant-weighted rotation regularizer α_b (Eq. 4–5)."""
    rng = np.random.default_rng(seed)
    ndesc = rng.integers(0, 8, size=num_bones)
    alpha = 1.0 + ndesc
    return alpha / alpha.mean()


def relative_depth_loss(
    z_canon: np.ndarray,
    z_deformed: np.ndarray,
    m_canon: np.ndarray,
    m_deformed: np.ndarray,
) -> float:
    """Closed-form scale-aligned MSE on depth deltas (App. C.3)."""
    dz = np.asarray(z_deformed) - np.asarray(z_canon)
    dm = np.asarray(m_deformed) - np.asarray(m_canon)
    scale = float(np.dot(dz, dm) / (np.dot(dm, dm) + 1e-12))
    resid = dz - scale * dm
    return float(np.mean(resid**2))


def arap_edge_loss(canon: np.ndarray, deformed: np.ndarray, edges: np.ndarray) -> float:
    """L_ARAP — preserve canonical k-NN edge lengths (App. C.3)."""
    c = np.asarray(canon, dtype=np.float64)
    d = np.asarray(deformed, dtype=np.float64)
    loss = 0.0
    for i, j in edges:
        lc = np.linalg.norm(c[i] - c[j])
        ld = np.linalg.norm(d[i] - d[j])
        loss += (ld - lc) ** 2
    return float(loss / max(len(edges), 1))


def phase1_step_stub(
    canonical: np.ndarray,
    weights: np.ndarray,
    bone_angles: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Single Phase-1 toy step: LBS + rgb proxy + depth + rotation reg."""
    rng = np.random.default_rng(seed)
    b = bone_angles.shape[0]
    T = np.stack([se3_from_rotation(rodrigues(bone_angles[j])) for j in range(b)])
    deformed = lbs_deform(canonical, weights, T)
    z0 = canonical[:, 2]
    zt = deformed[:, 2]
    m0 = z0 + rng.normal(0, 0.02, size=z0.shape)
    mt = zt + rng.normal(0, 0.02, size=zt.shape)
    l_depth = relative_depth_loss(z0, zt, m0, mt)
    l_rgb = float(np.mean((deformed - canonical) ** 2)) * 0.1
    alpha = descendant_weights(b, seed=seed)
    l_rot = float(np.mean(alpha * np.sum(bone_angles**2, axis=1)))
    return {
        "Lrgb": l_rgb,
        "L_delta_depth": l_depth,
        "Lrot": l_rot,
        "total": PHASE1_WEIGHTS["Lrgb"] * l_rgb
        + PHASE1_WEIGHTS["L_delta_depth"] * l_depth
        + PHASE1_WEIGHTS["Lrot"] * l_rot,
    }


def phase2_step_stub(
    canonical: np.ndarray,
    weight_logits: np.ndarray,
    bone_angles: np.ndarray,
    correction: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Phase-2 toy step with joint corrections + weight refinement."""
    weights = softmax_weights(weight_logits)
    composed = bone_angles + correction
    b = composed.shape[0]
    T = np.stack([se3_from_rotation(rodrigues(composed[j])) for j in range(b)])
    deformed = lbs_deform(canonical, weights, T)
    rng = np.random.default_rng(seed)
    edges = np.stack([rng.integers(0, len(canonical), size=2) for _ in range(32)])
    l_arap = arap_edge_loss(canonical, deformed, edges)
    l_corr = float(np.mean(correction**2))
    l_rgb = float(np.mean((deformed - canonical) ** 2)) * 0.08
    return {
        "Lrgb": l_rgb,
        "LARAP": l_arap,
        "Lcorr_l2": l_corr,
        "total": PHASE2_WEIGHTS["Lrgb"] * l_rgb
        + PHASE2_WEIGHTS["LARAP"] * l_arap
        + PHASE2_WEIGHTS["Lcorr_l2"] * l_corr,
    }


def optimization_bundle() -> dict[str, Any]:
    return {
        "phase1_weights": PHASE1_WEIGHTS,
        "phase2_weights": PHASE2_WEIGHTS,
        "phase1_iters_per_frame": 2000,
        "phase2_iters": 30_000,
        "sequential_warmstart": True,
        "depth_decay_phase2": True,
        "track_decay_phase2": True,
    }
