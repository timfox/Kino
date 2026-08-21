"""Extrinsic regression heads (Sec. III-C.3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.clrnet.se3 import quat_normalize


class ExtrinsicRegressionHead(nn.Module):
    """FC head: translation (3) + quaternion (4)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
        )
        self.trans = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.1), nn.Linear(256, 3))
        self.rot = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.1), nn.Linear(256, 4))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        code = self.shared(x)
        t = self.trans(code)
        q = quat_normalize(self.rot(code))
        return q, t


class SharedFeatureRegression(nn.Module):
    """Three heads on concatenated pairwise correlation features."""

    def __init__(self, pairwise_dim: int) -> None:
        super().__init__()
        shared_in = pairwise_dim * 3
        self.head_cl = ExtrinsicRegressionHead(shared_in)
        self.head_lr = ExtrinsicRegressionHead(shared_in)
        self.head_rc = ExtrinsicRegressionHead(shared_in)

    def forward(self, corr_cl: Tensor, corr_lr: Tensor, corr_rc: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
        shared = torch.cat([corr_cl, corr_lr, corr_rc], dim=-1)
        return {
            "CL": self.head_cl(shared),
            "LR": self.head_lr(shared),
            "RC": self.head_rc(shared),
        }

