"""PriOr-Flow dual-branch stub (Fig. 4, PriOr-RAFT)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.prior_flow.config import PriorFlowConfig
from ltx_trainer.prior_flow.dccl import build_correlation, dccl_lookup
from ltx_trainer.prior_flow.oddc import ODDCEncoder, group_wise_correlation
from ltx_trainer.prior_flow.orthogonal_view import (
    flow_orthogonal_to_primitive,
    orthogonal_to_primitive,
    primitive_to_orthogonal,
)


class _FeatureEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, dim, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _FlowGRU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.update = nn.Conv2d(dim * 2 + 2, dim, 3, padding=1)
        self.delta = nn.Conv2d(dim, 2, 3, padding=1)

    def forward(self, hidden: Tensor, motion: Tensor, flow: Tensor) -> tuple[Tensor, Tensor]:
        h = torch.tanh(self.update(torch.cat([hidden, motion, flow], dim=1)))
        return h, self.delta(h)


class PriorFlowStub(nn.Module):
    def __init__(self, cfg: PriorFlowConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PriorFlowConfig()
        d = self.cfg.feature_dim
        self.enc_p = _FeatureEncoder(d)
        self.enc_o = _FeatureEncoder(d)
        self.oddc = ODDCEncoder(d)
        self.motion_proj = nn.Conv2d(1, d, 1)
        self.gru_p = _FlowGRU(d)
        self.gru_o = _FlowGRU(d)

    def _pyramid(self, corr: Tensor) -> list[Tensor]:
        levels = [corr]
        for _ in range(3):
            levels.append(F.avg_pool2d(levels[-1], 2, stride=2))
        return levels

    def forward(self, frame1: Tensor, frame2: Tensor) -> dict[str, Tensor]:
        """frame1/frame2 [B,3,H,W] → primitive flow Fp and orthogonal Fo."""
        b, _, h, w = frame1.shape
        io1 = primitive_to_orthogonal(frame1)
        io2 = primitive_to_orthogonal(frame2)
        fp1, fp2 = self.enc_p(frame1), self.enc_p(frame2)
        fo1, fo2 = self.enc_o(io1), self.enc_o(io2)

        cp = self._pyramid(build_correlation(fp1, fp2))
        co = self._pyramid(build_correlation(fo1, fo2))
        _, _, fh, fw = fp1.shape

        flow_p = torch.zeros(b, 2, h, w, device=frame1.device)
        flow_o = torch.zeros(b, 2, h, w, device=frame1.device)
        hp = torch.zeros(b, self.cfg.feature_dim, fh, fw, device=frame1.device)
        ho = torch.zeros(b, self.cfg.feature_dim, fh, fw, device=frame1.device)
        preds_p: list[Tensor] = []
        preds_o: list[Tensor] = []

        for _ in range(self.cfg.num_iterations):
            if self.cfg.use_dccl:
                cp_i, co2p = dccl_lookup(cp, co, flow_p, radius=self.cfg.dccl_radius)
                co_i, cp2o = dccl_lookup(co, cp, flow_o, radius=self.cfg.dccl_radius)
            else:
                cp_i, co2p = cp[0], cp[0]
                co_i, cp2o = co[0], co[0]

            fo2p = flow_orthogonal_to_primitive(flow_o)
            flow_s = F.interpolate(flow_p, size=(fh, fw), mode="bilinear", align_corners=False)
            flow_s = torch.stack(
                [flow_s[:, 0] * (fw / w), flow_s[:, 1] * (fh / h)],
                dim=1,
            )
            if self.cfg.use_oddc:
                fo2p_s = F.interpolate(fo2p, size=(fh, fw), mode="bilinear", align_corners=False)
                fo2p_s = torch.stack(
                    [fo2p_s[:, 0] * (fw / w), fo2p_s[:, 1] * (fh / h)],
                    dim=1,
                )
                warped_p = F.grid_sample(
                    fp2,
                    self._flow_grid(flow_s, fh, fw),
                    align_corners=True,
                    padding_mode="border",
                )
                warped_o = F.grid_sample(
                    fp2,
                    self._flow_grid(fo2p_s, fh, fw),
                    align_corners=True,
                    padding_mode="border",
                )
                gp = group_wise_correlation(fp1, warped_p)
                go2p = group_wise_correlation(fp1, warped_o)
                mp = self.oddc(cp_i, co2p, gp, go2p, flow_s, fo2p_s)
            else:
                mp = self.motion_proj(cp_i)

            hp, dp = self.gru_p(hp, mp, flow_s)
            flow_p = flow_p + self._upsample_flow(dp, h, w)
            preds_p.append(flow_p)

            flow_o_s = F.interpolate(flow_o, size=(fh, fw), mode="bilinear", align_corners=False)
            flow_o_s = torch.stack(
                [flow_o_s[:, 0] * (fw / w), flow_o_s[:, 1] * (fh / h)],
                dim=1,
            )
            mo = self.motion_proj(co_i)
            ho, do = self.gru_o(ho, mo, flow_o_s)
            flow_o = flow_o + self._upsample_flow(do, h, w)
            preds_o.append(orthogonal_to_primitive(flow_o))

        return {
            "flow_primitive": flow_p,
            "flow_orthogonal": flow_o,
            "flow_orthogonal_in_primitive": flow_orthogonal_to_primitive(flow_o),
            "predictions_primitive": preds_p,
            "predictions_orthogonal": preds_o,
        }

    @staticmethod
    def _upsample_flow(delta: Tensor, h: int, w: int) -> Tensor:
        up = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)
        scale_w = w / delta.shape[-1]
        scale_h = h / delta.shape[-2]
        return torch.stack([up[:, 0] * scale_w, up[:, 1] * scale_h], dim=1)

    @staticmethod
    def _flow_grid(flow: Tensor, h: int, w: int) -> Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=flow.device),
            torch.linspace(-1, 1, w, device=flow.device),
            indexing="ij",
        )
        grid = torch.stack(
            [
                xx.unsqueeze(0).expand(flow.shape[0], -1, -1) + 2 * flow[:, 0] / max(w - 1, 1),
                yy.unsqueeze(0).expand(flow.shape[0], -1, -1) + 2 * flow[:, 1] / max(h - 1, 1),
            ],
            dim=-1,
        )
        return grid
