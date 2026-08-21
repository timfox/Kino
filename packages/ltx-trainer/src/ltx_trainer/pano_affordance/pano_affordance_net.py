"""PanoAffordanceNet end-to-end model (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.pano_affordance.config import PanoAffordanceConfig
from ltx_trainer.pano_affordance.dasm import DASM
from ltx_trainer.pano_affordance.encoders import TextEncoderStub, VisionEncoderStub
from ltx_trainer.pano_affordance.osdh import OSDH


class PanoAffordanceNet(nn.Module):
    def __init__(self, cfg: PanoAffordanceConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoAffordanceConfig()
        d = self.cfg.embed_dim
        self.vision = VisionEncoderStub(self.cfg)
        self.text = TextEncoderStub(self.cfg)
        self.dasm = DASM(d, d)
        self.decoder_cross = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.osdh = OSDH(top_k=self.cfg.osdh_top_k)

    def initial_maps(self, f_v: Tensor, f_t: Tensor) -> Tensor:
        """A_init from scaled dot-product (Eq. 4)."""
        sim = torch.bmm(f_t, f_v.transpose(1, 2))
        return sim * (f_v.shape[-1] ** -0.5)

    def region_pool(self, f_v: Tensor, heatmap: Tensor) -> Tensor:
        """vc from Eq. 9 — class-wise weighted pooling."""
        b, c, l = heatmap.shape
        weights = F.softmax(heatmap, dim=-1)
        pooled = torch.bmm(weights, f_v)
        return pooled

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        f_v = self.vision(image)
        f_t = self.text()
        f_t = f_t.expand(f_v.shape[0], -1, -1)
        f_vv = self.dasm(f_v, f_t)
        if f_vv.shape[1] != f_v.shape[1]:
            l = min(f_vv.shape[1], f_v.shape[1])
            f_vv = f_vv[:, :l]
        queries = f_t
        _out, _ = self.decoder_cross(queries, f_vv, f_vv)
        a_init = self.initial_maps(f_vv, queries)
        a_refined = self.osdh(f_vv, a_init)
        regions = self.region_pool(f_vv, a_refined.sigmoid())
        return {
            "a_init": a_init,
            "a_refined": a_refined,
            "region_feats": regions,
            "text_feats": queries,
            "vision_feats": f_vv,
        }
