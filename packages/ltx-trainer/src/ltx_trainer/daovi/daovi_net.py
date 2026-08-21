"""DAOVI framework stub (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.daovi.config import DaoviConfig
from ltx_trainer.daovi.gfcip import gfcip_propagate
from ltx_trainer.daovi.odafp import ODAFPBlock


class DaoviStub(nn.Module):
    def __init__(self, cfg: DaoviConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DaoviConfig()
        c = self.cfg.hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(4, c, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_enc = nn.Sequential(
            nn.Conv2d(1, c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, c // 2, 3, padding=1),
        )
        self.odafp = ODAFPBlock(c)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=c,
            nhead=4,
            dim_feedforward=c * 2,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1),
        )

    def forward(
        self,
        frames: Tensor,
        masks: Tensor,
        flow_fwd: Tensor,
        flow_bwd: Tensor,
        depth: Tensor,
    ) -> dict[str, Tensor]:
        """frames [B,T,3,H,W]; masks [B,T,1,H,W]; flow [B,T-1,2,H,W]; depth [B,T,1,H,W]."""
        b, t, _, h, w = frames.shape
        partial: list[Tensor] = []
        mr_maps: list[Tensor] = []
        for ti in range(t):
            if ti < t - 1 and self.cfg.use_gfcip:
                xp, mr = gfcip_propagate(
                    frames[:, ti],
                    frames[:, ti + 1],
                    masks[:, ti],
                    flow_fwd[:, ti],
                    flow_bwd[:, ti],
                    eps_deg=self.cfg.geodesic_eps_deg,
                )
            else:
                xp = frames[:, ti]
                mr = torch.zeros_like(masks[:, ti])
            partial.append(xp)
            mr_maps.append(mr)
        x_part = torch.stack(partial, dim=1)

        feats: list[Tensor] = []
        for ti in range(t):
            inp = torch.cat([x_part[:, ti], masks[:, ti]], dim=1)
            feat = self.encoder(inp)
            depth_f = F.interpolate(
                self.depth_enc(depth[:, ti]),
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.cfg.use_odafp and ti < t - 1:
                adj = self.encoder(torch.cat([x_part[:, ti + 1], masks[:, ti + 1]], dim=1))
                flow = F.interpolate(flow_fwd[:, ti], size=feat.shape[-2:], mode="bilinear", align_corners=False)
                unfilled = F.interpolate(1.0 - mr_maps[ti], size=feat.shape[-2:], mode="nearest")
                mask_s = F.interpolate(masks[:, ti], size=feat.shape[-2:], mode="nearest")
                feat = self.odafp(feat, adj, flow, depth_f, mask_s, unfilled, feat.shape[-2], feat.shape[-1])
            feats.append(feat)

        stacked = torch.stack(feats, dim=1)
        _, _, c, fh, fw = stacked.shape
        tokens = stacked.reshape(b * t, c, fh * fw).transpose(1, 2)
        tokens = self.transformer(tokens)
        feat = tokens.transpose(1, 2).view(b * t, c, fh, fw)
        recon = self.decoder(feat)
        if recon.shape[-2:] != (h, w):
            recon = F.interpolate(recon, size=(h, w), mode="bilinear", align_corners=False)
        recon = recon.view(b, t, 3, h, w)
        out = recon * masks + frames * (1.0 - masks)
        return {
            "frames": out,
            "partial": x_part,
            "reliability": torch.stack(mr_maps, dim=1),
        }
