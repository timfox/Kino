"""Smoke-scale fusion velocity head with parallel history modes."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.drfusion.adapter import ConditionAdapter
from ltx_trainer.drfusion.config import DRFusionConfig
from ltx_trainer.drfusion.history import (
    HistoryMode,
    build_history_window,
    stabilized_history_guidance,
)


class DRFusionVelocityHead(nn.Module):
    """Predict velocity under three history configurations; combine via Eq. (6)."""

    def __init__(self, channels: int = 8, cfg: DRFusionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DRFusionConfig()
        self.adapter = ConditionAdapter(in_channels=1, latent_channels=channels)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def _velocity(self, x: Tensor, c_struct: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        fused = []
        for i in range(t):
            fused.append(self.adapter.fuse_input(x[:, :, i], c_struct))
        x_in = torch.stack(fused, dim=2)
        return self.net(x_in)

    def forward(
        self,
        z_vis: Tensor,
        ir: Tensor,
        history: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """z_vis, history: (B, C, T, H, W); ir: (B, 1, H, W) current frame."""
        c_struct = self.adapter(ir)
        cfg = self.cfg
        vels: dict[str, Tensor] = {}
        for mode in (HistoryMode.BASELINE, HistoryMode.STABILIZED, HistoryMode.CONTEXT_SUPPRESS):
            h_win = build_history_window(history, mode, stabilize_sigma=cfg.stabilize_sigma)
            # use last frame of modulated history concatenated along time with current
            x = torch.cat([h_win[:, :, -1:], z_vis[:, :, -1:]], dim=2)
            vels[mode.value] = self._velocity(x, c_struct)
        v_guided = stabilized_history_guidance(
            vels["H0"][:, :, -1],
            vels["H1"][:, :, -1],
            vels["H2"][:, :, -1],
            cfg.guidance_scale,
        )
        return v_guided, vels
