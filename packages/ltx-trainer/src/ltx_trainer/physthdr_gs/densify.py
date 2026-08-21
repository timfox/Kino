"""Gaussian densification with illumination-guided gradient scaling (Eq. 17–18)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.physthdr_gs.gaussians import HDRGaussianField
from ltx_trainer.physthdr_gs.gradient_scaling import scaling_factor


def accumulate_viewspace_grad(
    field: HDRGaussianField,
    viewspace: Tensor,
    grad_scale: Tensor | None = None,
    *,
    state: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Accumulate |∂L/∂μ_ndc| for densification (Eq. 3)."""
    if viewspace.grad is None:
        return state or {"grad": torch.zeros(field.num_points, device=viewspace.device), "count": torch.zeros(1)}
    grad = viewspace.grad.detach()
    if grad.dim() == 2 and grad.shape[-1] >= 2:
        gnorm = grad[:, :2].norm(dim=-1)
    else:
        gnorm = grad.norm(dim=-1)
    if grad_scale is not None:
        gnorm = gnorm * grad_scale.squeeze(-1)
    st = state
    if st is None:
        st = {
            "grad": torch.zeros(field.num_points, device=gnorm.device),
            "count": torch.zeros(1, device=gnorm.device),
        }
    n = min(gnorm.shape[0], st["grad"].shape[0])
    st["grad"][:n] += gnorm[:n]
    st["count"] += 1
    return st


def densify_step(
    field: HDRGaussianField,
    grad_state: dict[str, Tensor],
    *,
    tau: float = 2e-4,
    scale_s: float = 1.0,
    la: Tensor | None = None,
    la_hat: Tensor | None = None,
    percent_dense: float = 0.02,
) -> int:
    """
    Clone Gaussians whose scaled gradient exceeds τ_p (Eq. 18).

    Returns number of points cloned.
    """
    if grad_state["count"].item() < 1:
        return 0
    avg_grad = grad_state["grad"] / grad_state["count"].clamp(min=1.0)
    if la is not None and la_hat is not None:
        sa = scaling_factor(la, la_hat, s=scale_s).squeeze(-1)
        if sa.shape[0] == avg_grad.shape[0]:
            avg_grad = avg_grad * sa
    mask = avg_grad > tau
    # Prefer under-reconstructed large Gaussians
    scales = field.get_scales().mean(dim=-1)
    mask = mask & (scales > percent_dense)
    n_before = field.num_points
    field.clone_points(mask)
    grad_state["grad"] = torch.zeros(field.num_points, device=avg_grad.device)
    grad_state["count"] = torch.zeros(1, device=avg_grad.device)
    return field.num_points - n_before
