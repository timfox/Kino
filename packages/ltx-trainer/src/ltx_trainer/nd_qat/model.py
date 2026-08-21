"""Lightweight neural distinguisher model stub."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.nd_qat.boolean_ops import BooleanConv2d, indicator
from ltx_trainer.nd_qat.config import NDQATConfig
from ltx_trainer.nd_qat.lsq import lsq_quantize


class GohrDistinguisher(nn.Module):
    """Original-style ND with standard conv layers (Sec. 1.1)."""

    def __init__(self, cfg: NDQATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NDQATConfig()
        c = self.cfg
        self.conv0 = nn.Conv2d(c.input_channels, c.conv0_out, kernel_size=1)
        self.conv1 = nn.Conv2d(c.residual_channels, c.residual_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c.residual_channels, c.residual_channels, kernel_size=3, padding=1)
        flat = c.residual_channels * c.input_height * c.group_size
        self.fc1 = nn.Linear(flat, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
        self.delta0 = nn.Parameter(torch.tensor(0.1))

    def _to_nchw(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return x.unsqueeze(-1)
        if x.shape[1] == self.cfg.input_channels:
            return x
        if x.shape[-1] == self.cfg.input_channels:
            return x.permute(0, 3, 1, 2)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._to_nchw(x)
        h = F.relu(self.conv0(x))
        h = F.relu(self.conv1(h))
        h = h + F.relu(self.conv2(h))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc_out(h)).squeeze(-1)

    def quantize_weights(self) -> dict[str, Tensor]:
        return {"conv0": lsq_quantize(self.conv0.weight, self.delta0)}


class Conv0Boolean(nn.Module):
    """Table 5: four active output channels via element-wise AND (Sec. 2)."""

    ACTIVE_CHANNELS = (1, 15, 24, 25)
    # (out_idx, left_in, right_in) for Cl=0, Cr=1, C'l=2, C'r=3
    EXPRESSIONS = (
        (1, 2, 0),   # C'_l ∧ Cl
        (15, 0, 2),  # Cl ∧ C'_l
        (24, 1, 3),  # Cr ∧ C'_r
        (25, 3, 1),  # C'_r ∧ Cr
    )

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        b, _, h, w = x.shape
        out = torch.zeros(b, self.out_channels, h, w, device=x.device, dtype=x.dtype)
        for oc, left, right in self.EXPRESSIONS:
            out[:, oc] = x[:, left] * x[:, right]
        return out


class LightweightDistinguisher(nn.Module):
    """QAT lightweight ND with Boolean conv0 + standard tail stub."""

    def __init__(self, cfg: NDQATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NDQATConfig()
        self.gohr = GohrDistinguisher(self.cfg)
        self.conv0_bool = Conv0Boolean(self.cfg.conv0_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.gohr._to_nchw(x)
        h = self.conv0_bool(x)
        h = F.relu(self.gohr.conv1(h))
        h = h + F.relu(self.gohr.conv2(h))
        h = h.flatten(1)
        h = F.relu(self.gohr.fc1(h))
        h = F.relu(self.gohr.fc2(h))
        return torch.sigmoid(self.gohr.fc_out(h)).squeeze(-1)

    def predict_label(self, prob: Tensor) -> Tensor:
        return (prob >= self.cfg.threshold).float()
