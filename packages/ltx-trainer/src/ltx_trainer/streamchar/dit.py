"""Joint audio-video DiT smoke model (Eq. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.streamchar.config import StreamCharConfig
from ltx_trainer.streamchar.flow import flow_matching_loss


class StreamCharDiT(nn.Module):
    """Concatenate [z_ref, z_mot, x_v^t, x_a^t] and predict velocities."""

    def __init__(self, channels: int = 8, cfg: StreamCharConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or StreamCharConfig()
        c = channels
        self.encoder = nn.Sequential(
            nn.Conv3d(c * 4, c * 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(c * 2, c * 2, 3, padding=1),
        )
        self.v_head = nn.Conv3d(c * 2, c, 3, padding=1)
        self.a_head = nn.Conv3d(c * 2, c, 3, padding=1)

    @staticmethod
    def _match_temporal(x: Tensor, t_len: int) -> Tensor:
        if x.shape[2] == t_len:
            return x
        if x.shape[2] == 1:
            return x.expand(-1, -1, t_len, -1, -1)
        return torch.nn.functional.interpolate(
            x, size=(t_len, x.shape[3], x.shape[4]), mode="trilinear", align_corners=False
        )

    def forward(
        self,
        z_ref: Tensor,
        z_mot: Tensor,
        x_v: Tensor,
        x_a: Tensor,
        c_a: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """All video tensors (B, C, T, H, W); audio (B, C, T_a, H, W) — T_a matched for smoke."""
        if x_a.shape[2] != x_v.shape[2]:
            x_a = torch.nn.functional.interpolate(
                x_a, size=x_v.shape[2:], mode="trilinear", align_corners=False
            )
        if c_a.dim() == 3:
            ca = c_a.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand_as(x_a)
        else:
            ca = c_a
        x_a = x_a + ca
        t_len = x_v.shape[2]
        z_ref = self._match_temporal(z_ref, t_len)
        z_mot = self._match_temporal(z_mot, t_len)
        tokens = torch.cat([z_ref, z_mot, x_v, x_a], dim=1)
        h = self.encoder(tokens)
        return self.v_head(h), self.a_head(h)

    def training_loss(
        self,
        z_v: Tensor,
        z_a: Tensor,
        z_ref: Tensor,
        z_mot: Tensor,
        c_a: Tensor,
        t: Tensor,
    ) -> Tensor:
        from ltx_trainer.streamchar.flow import corrupt_latents

        x_v, x_a, eps_v, eps_a = corrupt_latents(z_v, z_a, t)
        pred_v, pred_a = self.forward(z_ref, z_mot, x_v, x_a, c_a)
        return flow_matching_loss(pred_v, pred_a, z_v, z_a, eps_v, eps_a)
