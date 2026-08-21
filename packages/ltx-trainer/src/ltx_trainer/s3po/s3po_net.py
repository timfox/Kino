"""S3PO recurrent 360° VSR stub (Fig. 4–8)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.s3po.config import S3POConfig
from ltx_trainer.s3po.cyclic import cyclic_shift_erp
from ltx_trainer.s3po.dual_duct import DualDuctStack
from ltx_trainer.s3po.feature_extractor import PanoramaFeatureExtractor


class S3POStub(nn.Module):
    """Sliding-window recurrent 360° ×4 VSR without explicit alignment."""

    def __init__(self, cfg: S3POConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or S3POConfig()
        d = self.cfg.feature_dim
        s = self.cfg.scale
        self.extractor = PanoramaFeatureExtractor(d, use_attention=self.cfg.use_attention)
        self.global_fuse = nn.Conv2d(3 + d + 3 * s * s, d, 3, padding=1)
        self.ducts = DualDuctStack(d, self.cfg.num_duct_blocks)
        self.up_local = nn.Sequential(
            nn.Conv2d(d, 3 * s * s, 3, padding=1),
            nn.PixelShuffle(s),
        )
        self.up_global = nn.Sequential(
            nn.Conv2d(d, 3 * s * s, 3, padding=1),
            nn.PixelShuffle(s),
        )
        self.res_fuse = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def _global_path(
        self,
        target_lr: Tensor,
        hidden: Tensor | None,
        prev_hr: Tensor | None,
    ) -> Tensor:
        b, _, h, w = target_lr.shape
        parts = [target_lr]
        if hidden is None:
            hidden = torch.zeros(b, self.cfg.feature_dim, h, w, device=target_lr.device)
        parts.append(hidden)
        if prev_hr is None:
            prev_hr = F.interpolate(target_lr, scale_factor=self.cfg.scale, mode="bilinear", align_corners=False)
        prev_ds = F.pixel_unshuffle(prev_hr, self.cfg.scale)
        parts.append(prev_ds)
        fused = torch.cat(parts, dim=1)
        return F.relu(self.global_fuse(fused))

    def _local_path(self, prev_f: Tensor, target: Tensor, next_f: Tensor) -> Tensor:
        g = self.extractor(prev_f, target, next_f)
        if self.cfg.use_cyclic:
            g_cyc = self.extractor(
                cyclic_shift_erp(prev_f),
                cyclic_shift_erp(target),
                cyclic_shift_erp(next_f),
            )
            g = g + g_cyc
        return g

    def forward(self, clip: Tensor) -> dict[str, Tensor]:
        """
        clip [B,T,3,H,W] low-res ERP sequence → HR frames [B,T,3,sH,sW].
        """
        b, t, c, h, w = clip.shape
        assert c == 3
        hidden: Tensor | None = None
        prev_hr: Tensor | None = None
        hr_list: list[Tensor] = []

        for ti in range(t):
            prev_f = clip[:, max(ti - 1, 0)]
            target = clip[:, ti]
            next_f = clip[:, min(ti + 1, t - 1)]
            glocal = self._local_path(prev_f, target, next_f)
            gglobal = self._global_path(target, hidden, prev_hr)
            lf, gf, hidden = self.ducts(glocal, gglobal)
            res_l = self.up_local(lf)
            res_g = self.up_global(gf)
            residue = self.res_fuse(torch.cat([res_l, res_g], dim=1))
            base = F.interpolate(target, scale_factor=self.cfg.scale, mode="bilinear", align_corners=False)
            hr = (base + residue).clamp(0.0, 1.0)
            hr_list.append(hr)
            prev_hr = hr.detach()

        hr_seq = torch.stack(hr_list, dim=1)
        return {
            "hr": hr_seq,
            "last_hidden": hidden,
            "scale": torch.tensor(float(self.cfg.scale)),
        }
