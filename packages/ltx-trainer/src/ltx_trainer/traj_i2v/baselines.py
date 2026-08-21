"""RIFE and optical-flow reconstruction baselines (Sec. III-A)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def warp_frame(
    frame: Tensor,
    flow: Tensor,
) -> Tensor:
    """Backward-warp ``frame`` [B,3,H,W] with flow [B,2,H,W] (pixel displacements)."""
    b, _, h, w = frame.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=frame.device, dtype=frame.dtype),
        torch.arange(w, device=frame.device, dtype=frame.dtype),
        indexing="ij",
    )
    gx = xx + flow[:, 0]
    gy = yy + flow[:, 1]
    gx = 2.0 * gx / max(w - 1, 1) - 1.0
    gy = 2.0 * gy / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


def optical_flow_extrapolate(
    reference: Tensor,
    *,
    num_frames: int,
    flow_scale: float = 0.15,
) -> Tensor:
    """Farneback-style stub: constant forward flow scaled per frame index."""
    if reference.dim() == 3:
        reference = reference.unsqueeze(0)
    b, c, h, w = reference.shape
    try:
        import cv2  # noqa: F401

        ref = reference[0].permute(1, 2, 0).detach().cpu().numpy()
        gray = (0.299 * ref[..., 0] + 0.587 * ref[..., 1] + 0.114 * ref[..., 2]).astype("float32")
        flow = cv2.calcOpticalFlowFarneback(gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        base_flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(reference.device)
    except Exception:
        base_flow = torch.zeros(b, 2, h, w, device=reference.device, dtype=reference.dtype)
        base_flow[:, 0] = flow_scale * w * 0.01
        base_flow[:, 1] = -flow_scale * h * 0.005

    frames: list[Tensor] = []
    for i in range(num_frames):
        t = (i + 1) / max(num_frames, 1)
        flow_t = base_flow * t
        frames.append(warp_frame(reference, flow_t))
    return torch.cat(frames, dim=0)


def rife_interpolate_stub(
    frame0: Tensor,
    frame1: Tensor,
    *,
    num_frames: int,
) -> Tensor:
    """Practical-RIFE placeholder: endpoint linear blend (bidirectional flow not bundled)."""
    if frame0.dim() == 3:
        frame0 = frame0.unsqueeze(0)
    if frame1.dim() == 3:
        frame1 = frame1.unsqueeze(0)
    outs: list[Tensor] = []
    for i in range(num_frames):
        t = (i + 1) / (num_frames + 1)
        outs.append((1 - t) * frame0 + t * frame1)
    return torch.cat(outs, dim=0)
