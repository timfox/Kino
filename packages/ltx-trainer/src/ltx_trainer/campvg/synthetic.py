"""Synthetic panoramic clips for CPU smoke."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.campvg.config import CamPVGConfig


def synthetic_poses(cfg: CamPVGConfig, batch: int = 1) -> tuple[Tensor, Tensor]:
    """Random SO(3) trajectories with small translations."""
    n = cfg.num_frames
    rots = []
    trans = []
    for _ in range(batch):
        r_list = []
        t_list = []
        r_acc = torch.eye(3)
        t_acc = torch.zeros(3)
        for fi in range(n):
            yaw = 0.05 * fi
            r_yaw = torch.tensor(
                [
                    [torch.cos(torch.tensor(yaw)), 0.0, torch.sin(torch.tensor(yaw))],
                    [0.0, 1.0, 0.0],
                    [-torch.sin(torch.tensor(yaw)), 0.0, torch.cos(torch.tensor(yaw))],
                ]
            )
            r_acc = r_yaw @ r_acc
            t_acc = t_acc + torch.tensor([0.02, 0.0, 0.0])
            r_list.append(r_acc)
            t_list.append(t_acc.clone())
        rots.append(torch.stack(r_list))
        trans.append(torch.stack(t_list))
    return torch.stack(rots), torch.stack(trans)


def synthetic_erp_video(cfg: CamPVGConfig, batch: int = 1) -> Tensor:
    """[B, N, 3, H, W] gradient ERP."""
    h, w, n = cfg.height, cfg.width, cfg.num_frames
    v = torch.linspace(0, 1, h).view(h, 1, 1)
    u = torch.linspace(0, 1, w).view(1, w, 1)
    base = torch.cat([u.expand(h, w, 1), v.expand(h, w, 1), (u * v).expand(h, w, 1)], dim=-1)
    base = base.permute(2, 0, 1)
    frames = []
    for fi in range(n):
        frames.append(base * (0.9 + 0.02 * fi))
    vid = torch.stack(frames, dim=0)
    return vid.unsqueeze(0).expand(batch, n, 3, h, w).clone()
