"""Motion drift and motion field heatmaps (Appendix A)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.lamo.latent_motion import latent_delta, macro_drift_vector, select_strongest_motion_index


def motion_drift_heatmap(
    z: Tensor,
    *,
    tau: int = 2,
    t_star: int | None = None,
    epsilon: float = 1e-6,
) -> tuple[Tensor, int]:
    """R_drift(h,w) = |⟨Δ_{t*}, b_{t*}⟩| / (||b_{t*}|| + ε) (Eq. 12)."""
    if z.dim() != 4:
        raise ValueError("z must be (T, C, H, W)")
    t_star = t_star if t_star is not None else select_strongest_motion_index(z, tau=tau, dim=0)
    delta = latent_delta(z, tau=tau, dim=0)
    delta_t = delta[t_star]
    bt = macro_drift_vector(delta_t.unsqueeze(0)).squeeze(0)
    bt_norm = bt.norm() + epsilon
    # per-pixel projection onto macro drift direction
    c, h, w = delta_t.shape
    flat = delta_t.reshape(c, -1)
    proj = (flat * bt[:, None]).sum(dim=0).abs() / bt_norm
    return proj.reshape(h, w), t_star


def motion_field_heatmap(
    z: Tensor,
    predictor: torch.nn.Module,
    cond: Tensor,
    *,
    tau: int = 2,
    t_star: int | None = None,
) -> tuple[Tensor, int]:
    """R_field from f_φ(z(t*),c) − f_φ(z_static,c) (Eq. 14)."""
    if z.dim() != 4:
        raise ValueError("z must be (T, C, H, W)")
    t_star = t_star if t_star is not None else select_strongest_motion_index(z, tau=tau, dim=0)
    z_static = z.mean(dim=0, keepdim=True)
    with torch.no_grad():
        resp_star = predictor(z[t_star].unsqueeze(0), cond)
        resp_static = predictor(z_static, cond)
    diff = resp_star - resp_static
    return diff.squeeze(0).norm(dim=0), t_star
