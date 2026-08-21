"""Inverse-kinematics and 4D generation metrics (Tables 1–2)."""

from __future__ import annotations

import torch
from torch import Tensor


def chamfer_l1(pred: Tensor, target: Tensor) -> float:
    """Symmetric Chamfer L1 on N×3 point sets."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must share shape (N, 3)")
    d = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0), p=1)[0]
    return float((d.min(dim=0).values.mean() + d.min(dim=1).values.mean()).item() / 2.0)


def chamfer_l2(pred: Tensor, target: Tensor) -> float:
    d = torch.cdist(pred.unsqueeze(0), target.unsqueeze(0), p=2)[0]
    return float((d.min(dim=0).values.mean() + d.min(dim=1).values.mean()).item() / 2.0)


def voxel_iou(pred: Tensor, target: Tensor, *, grid: int = 16) -> float:
    """Occupancy IoU by voxelizing axis-aligned bounds (PartNet-Mobility eval proxy)."""
    pts = torch.cat([pred, target], dim=0)
    lo, hi = pts.min(0).values, pts.max(0).values
    span = (hi - lo).clamp(min=1e-3)

    def _vox(x: Tensor) -> Tensor:
        idx = ((x - lo) / span * (grid - 1)).long().clamp(0, grid - 1)
        key = idx[:, 0] * grid * grid + idx[:, 1] * grid + idx[:, 2]
        occ = torch.zeros(grid**3, dtype=torch.bool)
        occ[key] = True
        return occ

    a, b = _vox(pred), _vox(target)
    inter = (a & b).sum().float()
    union = (a | b).sum().float().clamp(min=1.0)
    return float((inter / union).item())


def inverse_kinematics_smoke(
    *,
    num_points: int = 64,
    latent_dim: int = 16,
    steps: int = 40,
    seed: int = 7,
) -> dict[str, float]:
    """Optimize latent z so decoded point cloud matches a target pose."""
    torch.manual_seed(seed)
    target = torch.randn(num_points, 3)
    w = torch.randn(num_points * 3, latent_dim) * 0.1
    z = torch.zeros(latent_dim, requires_grad=True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(steps):
        pred = (z @ w.T).view(num_points, 3)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pred = (z.detach() @ w.T).view(num_points, 3)
    return {
        "chamfer_l1": round(chamfer_l1(pred, target), 4),
        "chamfer_l2": round(chamfer_l2(pred, target), 4),
        "iou": round(voxel_iou(pred, target), 4),
        "final_mse": round(float(loss.detach()), 4),
    }
