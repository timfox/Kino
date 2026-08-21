"""Progressive streaming simulation (Fig. 10–11)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.evogs.metrics import ssim_proxy


def transmit_order_by_psi_energy(psi_norms: list[float]) -> list[int]:
    """EvoGS prioritizes higher ||ψ|| refinements first (Sec. 4.3)."""
    return sorted(range(len(psi_norms)), key=lambda i: psi_norms[i], reverse=True)


def transmit_order_by_opacity(opacities: list[float]) -> list[int]:
    """LapisGS-style descending opacity order."""
    return sorted(range(len(opacities)), key=lambda i: opacities[i], reverse=True)


def progressive_curve(
    base_ssim: float,
    final_ssim: float,
    n_steps: int,
    *,
    smooth: bool = True,
) -> list[float]:
    """Simulate SSIM vs received fraction for continuous vs discrete layering."""
    xs = np.linspace(0, 1, n_steps)
    if smooth:
        # Monotone smooth ramp (EvoGS)
        return [float(base_ssim + (final_ssim - base_ssim) * (1 - np.exp(-4 * t))) for t in xs]
    # Discrete layering: flat plateaus with jumps at LOD boundaries
    out: list[float] = []
    lod_bounds = [0.25, 0.5, 0.75, 1.0]
    lod_ssim = [base_ssim, base_ssim + 0.05, base_ssim + 0.12, final_ssim - 0.02, final_ssim]
    li = 0
    for t in xs:
        while li + 1 < len(lod_bounds) and t > lod_bounds[li]:
            li += 1
        out.append(float(lod_ssim[li]))
    return out


def quality_transition_metrics(
    continuous: list[float],
    discrete: list[float],
) -> dict[str, Any]:
    dc = np.diff(continuous)
    dd = np.diff(discrete)
    return {
        "continuous_monotone": bool(np.all(dc >= -1e-6)),
        "continuous_max_jump": float(np.max(np.abs(dc))) if dc.size else 0.0,
        "discrete_max_jump": float(np.max(np.abs(dd))) if dd.size else 0.0,
        "continuous_final_ssim": float(continuous[-1]),
        "discrete_final_ssim": float(discrete[-1]),
        "smoothness_gain": float(np.max(np.abs(dd)) / max(np.max(np.abs(dc)), 1e-6)),
    }


def streaming_demo(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gt = rng.random(64)
    pred_coarse = gt + rng.normal(scale=0.08, size=gt.shape)
    pred_fine = gt + rng.normal(scale=0.02, size=gt.shape)
    base = ssim_proxy(gt, pred_coarse)
    final = ssim_proxy(gt, pred_fine)
    cont = progressive_curve(base, final, 20, smooth=True)
    disc = progressive_curve(base, final, 20, smooth=False)
    return {
        "quality_metrics": quality_transition_metrics(cont, disc),
        "continuous_curve": cont,
        "discrete_curve": disc,
    }
