"""VU-BOIQA network stub (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.vuboiqa.aps import crop_patches
from ltx_trainer.vuboiqa.config import VuBoiqaConfig
from ltx_trainer.vuboiqa.lgqa import LGQAModule
from ltx_trainer.vuboiqa.losses import norm_in_norm_loss
from ltx_trainer.vuboiqa.pdff import PDFFModule


class SwinBackboneStub(nn.Module):
    """Frozen Swin-V2 stub — multi-stage features (Eq. 3)."""

    def __init__(self, ch: int = 64) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(3, ch, 4, stride=4), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ch, ch, 2, stride=2), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ch, ch, 2, stride=2), nn.ReLU()),
                nn.Sequential(nn.Conv2d(ch, ch, 2, stride=2), nn.ReLU()),
            ]
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        feats: list[Tensor] = []
        h = x
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


class VuBoiqaStub(nn.Module):
    def __init__(self, cfg: VuBoiqaConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or VuBoiqaConfig()
        ch = 64
        self.backbone = SwinBackboneStub(ch)
        self.pdff = PDFFModule(ch) if self.cfg.use_pdff else nn.Identity()
        self.lgqa = LGQAModule(ch, self.cfg.embed_dim, self.cfg.num_heads)

    def _encode_patches(self, patches: Tensor) -> tuple[Tensor, Tensor]:
        b, k, c, p, _ = patches.shape
        patch_feats = []
        global_last = None
        for i in range(k):
            feats = self.backbone(patches[:, i])
            global_last = feats[-1]
            if self.cfg.use_pdff:
                fused = self.pdff(feats[:3])
            else:
                fused = feats[-1]
            patch_feats.append(fused)
        stacked = torch.stack(patch_feats, dim=1)
        assert global_last is not None
        return stacked, global_last

    def forward(
        self,
        erp: Tensor,
        *,
        mos: Tensor | None = None,
    ) -> dict[str, Tensor]:
        patches = crop_patches(erp, self.cfg)
        patch_feats, global_feat = self._encode_patches(patches)
        score = self.lgqa(patch_feats, global_feat)
        out: dict[str, Tensor] = {"quality": score}
        if mos is not None:
            out["loss"] = norm_in_norm_loss(score, mos.view_as(score), gamma=self.cfg.loss_gamma)
        return out
