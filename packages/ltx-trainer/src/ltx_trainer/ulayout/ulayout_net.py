"""uLayout unified layout network stub."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.ulayout.config import ULayoutConfig
from ltx_trainer.ulayout.erp import crop_informative_columns, pad_to_pano_width, vertical_shift_rows
from ltx_trainer.ulayout.extractor import DualBranchExtractor
from ltx_trainer.ulayout.losses import pano_loss, perspective_loss, total_loss
from ltx_trainer.ulayout.swg_transformer import SWGTransformer


class ULayoutStub(nn.Module):
    def __init__(self, cfg: ULayoutConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ULayoutConfig()
        self.extractor = DualBranchExtractor(self.cfg)
        ch = 64
        self.swg = SWGTransformer(ch, repeats=self.cfg.swg_repeats)
        self.head_ceiling = nn.Conv2d(ch, 1, kernel_size=1)
        self.head_floor = nn.Conv2d(ch, 1, kernel_size=1)
        self.depth_head = nn.Linear(self.cfg.pano_feature_w, self.cfg.pano_feature_w)

    def _prep_pano(self, x: Tensor) -> Tensor:
        return x

    def _prep_pp(self, x: Tensor, pitch_deg: float) -> Tensor:
        if self.cfg.use_vertical_shift:
            x = vertical_shift_rows(x, pitch_deg, x.shape[-2])
        return crop_informative_columns(x, self.cfg.pp_width)

    def forward(
        self,
        pano: Tensor,
        pp: Tensor,
        *,
        pitch_deg: float = 0.0,
        tgt_pano_b: Tensor | None = None,
        tgt_pp_b: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pano_in = self._prep_pano(pano)
        pp_in = self._prep_pp(pp, pitch_deg)
        fp, fpp = self.extractor(pano_in, pp_in)
        fpp_pad = pad_to_pano_width(fpp, fp.shape[-1])
        fused = self.swg(fp + fpp_pad)
        ceil = torch.sigmoid(self.head_ceiling(fused)).squeeze(1)
        floor = torch.sigmoid(self.head_floor(fused)).squeeze(1)
        depth = self.depth_head(ceil.mean(dim=1))
        out: dict[str, Tensor] = {
            "ceiling": ceil,
            "floor": floor,
            "depth": depth,
        }
        if tgt_pano_b is not None and tgt_pp_b is not None:
            tgt_c, tgt_f = tgt_pano_b[:, 0], tgt_pano_b[:, 1]

            def _resize_1d(pred: Tensor, tgt: Tensor) -> Tensor:
                if pred.shape == tgt.shape:
                    return pred
                return F.interpolate(
                    pred.unsqueeze(1), size=tgt.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)

            ceil_m = _resize_1d(ceil, tgt_c)
            depth_m = F.interpolate(
                depth.unsqueeze(1), size=tgt_f.shape[-1:], mode="linear", align_corners=False
            ).squeeze(1)
            lp = pano_loss(
                ceil_m,
                tgt_c,
                depth_m,
                tgt_f.mean(dim=1) if tgt_f.dim() == 3 else tgt_f,
                lambda_b=self.cfg.lambda_b,
                mu_d=self.cfg.mu_depth,
                gamma_g=self.cfg.gamma_geom,
            )
            ceil_pp = _resize_1d(ceil, tgt_pp_b)
            lpp = perspective_loss(ceil_pp, tgt_pp_b, delta=self.cfg.delta_pp)
            out["loss"] = total_loss(lp, lpp)
        return out
