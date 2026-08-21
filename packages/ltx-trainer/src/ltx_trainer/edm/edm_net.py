"""EDM dense matching network stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.edm.config import EdmConfig
from ltx_trainer.edm.encoder import MultiScaleEncoder
from ltx_trainer.edm.geodesic_refine import GeodesicRefiner
from ltx_trainer.edm.losses import angular_regression_loss, certainty_bce, total_loss
from ltx_trainer.edm.spherical import erp_grid_to_spherical
from ltx_trainer.edm.ssam import SSAModule


class EdmStub(nn.Module):
    def __init__(self, cfg: EdmConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EdmConfig()
        ch = self.cfg.feature_ch
        self.encoder = MultiScaleEncoder(ch)
        self.ssam = SSAModule(self.cfg)
        self.refiner = GeodesicRefiner(ch)

    def forward(
        self,
        img_a: Tensor,
        img_b: Tensor,
        *,
        tgt_s: Tensor | None = None,
        certainty_gt: Tensor | None = None,
    ) -> dict[str, Tensor]:
        feat_a_c, _ = self.encoder(img_a)
        feat_b_c, _ = self.encoder(img_b)
        s_coarse, c_coarse = self.ssam(feat_a_c, feat_b_c)
        _, _, h, w = feat_a_c.shape
        u_grid = erp_grid_to_spherical(h, w, img_a.device).unsqueeze(0).expand(img_a.shape[0], -1, -1, -1)
        if self.cfg.use_geodesic_refine:
            s_match, _ = self.refiner(feat_a_c, feat_b_c, s_coarse, u_grid)
        else:
            s_match = s_coarse
        out: dict[str, Tensor] = {
            "match_s": s_match,
            "certainty": c_coarse,
        }
        if tgt_s is not None and certainty_gt is not None:
            c_pred = c_coarse.squeeze(1)
            if c_pred.shape != certainty_gt.shape:
                certainty_gt = torch.nn.functional.interpolate(
                    certainty_gt.unsqueeze(1), size=c_pred.shape[-2:], mode="nearest"
                ).squeeze(1)
            if s_match.shape != tgt_s.shape:
                tgt_s = torch.nn.functional.interpolate(
                    tgt_s.permute(0, 3, 1, 2), size=s_match.shape[1:3], mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                s_match_n = s_match
            else:
                s_match_n = s_match
            lr = angular_regression_loss(s_match_n, tgt_s, certainty_gt)
            lc = certainty_bce(c_pred, certainty_gt)
            out["loss"] = total_loss(lr, lc, lam=self.cfg.loss_lambda_c)
        return out
