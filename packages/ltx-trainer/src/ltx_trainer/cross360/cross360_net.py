"""Cross360 network stub (Fig. 1): ERP encoder + CPFA pyramid + PFAA depth head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.cross360.config import NUM_SCALES, Cross360Config
from ltx_trainer.cross360.cpfa import CrossProjectionFeatureAlignment
from ltx_trainer.cross360.erp_encoder import ERPEncoderStub
from ltx_trainer.cross360.pfaa import ProgressiveFeatureAggregationAttention
from ltx_trainer.cross360.tangent import sample_tp_patches_from_erp


class Cross360NetStub(nn.Module):
    def __init__(self, cfg: Cross360Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or Cross360Config()
        self.encoder = ERPEncoderStub(self.cfg)
        chs = self.encoder.out_channels
        out_c = chs[0]
        self.cpfa = nn.ModuleList(
            [CrossProjectionFeatureAlignment(chs[i], self.cfg.num_heads) for i in range(NUM_SCALES)]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(chs[i] * 2, out_c, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for i in range(NUM_SCALES)
            ]
        )
        self.depth_heads = nn.ModuleList([nn.Conv2d(out_c, 1, 1) for _ in range(NUM_SCALES)])
        self.pfaa = ProgressiveFeatureAggregationAttention([out_c] * NUM_SCALES)
        self.final_depth = nn.Conv2d(out_c, 1, 1)

    def forward(self, erp: Tensor) -> dict[str, Tensor]:
        b, _, h, w = erp.shape
        patch_size = max(16, min(h, w) // 8)
        tp = sample_tp_patches_from_erp(
            erp, patch_size=patch_size, incomplete_fov=self.cfg.incomplete_fov
        )
        f_erps = self.encoder(erp)

        decoded: list[Tensor] = []
        prev_dec: Tensor | None = None
        for s in range(NUM_SCALES):
            f_erp = f_erps[s]
            # Stub: RGB tangent patches at all scales (full pipeline re-samples from F_CA via ERP2TP).
            f_ca = self.cpfa[s](f_erp, tp)
            dec = self.decoder[s](torch.cat([f_erp, f_ca], dim=1))
            if prev_dec is not None:
                prev_up = F.interpolate(prev_dec, size=dec.shape[-2:], mode="bilinear", align_corners=False)
                dec = dec + prev_up
            decoded.append(dec)
            prev_dec = dec

        depth_scales = [self.depth_heads[s](decoded[s]) for s in range(NUM_SCALES)]
        depth_final = self.final_depth(self.pfaa(decoded))
        depth_final = F.interpolate(depth_final, size=(h, w), mode="bilinear", align_corners=False)

        return {
            "depth": depth_final,
            "depth_scales": depth_scales,
            "decoded": decoded,
        }
