"""Geodesic Flow-Consistent Image Propagation (Sec. 3.2, Eq. 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.daovi.geodesic import flow_consistency_error


def warp_frame(frame: Tensor, flow: Tensor) -> Tensor:
    """Backward warp frame with flow [B,2,H,W] (dx, dy)."""
    b, c, h, w = frame.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=frame.device),
        torch.linspace(-1, 1, w, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack(
        [
            xx.unsqueeze(0).expand(b, -1, -1) + 2 * flow[:, 0] / max(w - 1, 1),
            yy.unsqueeze(0).expand(b, -1, -1) + 2 * flow[:, 1] / max(h - 1, 1),
        ],
        dim=-1,
    )
    return F.grid_sample(frame, grid, align_corners=True, padding_mode="border")


def gfcip_propagate(
    frame_t: Tensor,
    frame_next: Tensor,
    mask_t: Tensor,
    flow_fwd: Tensor,
    flow_bwd: Tensor,
    *,
    eps_deg: float = 0.4,
) -> tuple[Tensor, Tensor]:
    """
    Eq. (1): X't = W(Xt+1, F) * Mr + Xt * (1 - Mr).

    Returns partially inpainted frame and reliability mask Mr.
    """
    valid_flow, _ = flow_consistency_error(flow_fwd, flow_bwd, eps_deg=eps_deg)
    warped = warp_frame(frame_next, flow_fwd)
    mr = valid_flow * mask_t
    x_prime = warped * mr + frame_t * (1.0 - mr)
    return x_prime, mr
