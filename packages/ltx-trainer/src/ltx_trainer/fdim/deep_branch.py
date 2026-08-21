"""Deep feature branch (Sec. III-B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.fdim.backbone import MultiScaleBackbone
from ltx_trainer.fdim.cafm import CAFM
from ltx_trainer.fdim.msf import MultiScaleFusion


class DeepBranch(nn.Module):
    def __init__(self, *, widths: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.backbone = MultiScaleBackbone(widths=widths)
        self.cafms = nn.ModuleList([CAFM(w) for w in widths])
        self.msf = MultiScaleFusion(widths)
        dim = sum(widths)
        self.head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.uncertainty = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1), nn.Softplus())

    def forward_frame(self, ref: Tensor, dist: Tensor) -> tuple[Tensor, Tensor]:
        if ref.dim() == 3:
            ref = ref.unsqueeze(0)
            dist = dist.unsqueeze(0)
        ref_feats = self.backbone(ref)
        dist_feats = self.backbone(dist)
        cafm_out = [m(r, d) for m, r, d in zip(self.cafms, ref_feats, dist_feats, strict=True)]
        v = self.msf(cafm_out)
        q = self.head(v)
        sigma = self.uncertainty(v) + 1e-3
        return q.squeeze(-1), sigma.squeeze(-1)

    def forward(self, ref: Tensor, dist: Tensor) -> tuple[Tensor, Tensor]:
        """``ref/dist`` ``[B,3,H,W]`` or ``[T,3,H,W]`` → ``(q, sigma)``."""
        if ref.dim() == 4 and ref.shape[1] == 3:
            qs, sigs = [], []
            for i in range(ref.shape[0]):
                q, s = self.forward_frame(ref[i], dist[i])
                qs.append(q if q.ndim else q.reshape(1))
                sigs.append(s if s.ndim else s.reshape(1))
            q_out = torch.cat(qs)
            s_out = torch.cat(sigs)
            return q_out.mean().reshape(1), s_out.mean().reshape(1)
        return self.forward_frame(ref, dist)
