"""TPGS two-stage Gaussian optimization stub (Fig. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.tpgs.config import TpgsConfig
from ltx_trainer.tpgs.cubemap import (
    FACE_NAMES,
    cartesian_to_spherical,
    face_grid,
    padded_fov_rad,
    spherical_to_erp_uv,
)
from ltx_trainer.tpgs.losses import photometric_loss, stitch_erp
from ltx_trainer.tpgs.transition_plane import view_rotations


class TpgsStub(nn.Module):
    """Perspective 3DGS on cube + transition views, then ERP fine-tune."""

    def __init__(self, cfg: TpgsConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TpgsConfig()
        n = self.cfg.cube_face_res
        self.gaussians = nn.Parameter(torch.randn(n * n, 3) * 0.1)
        self._erp_gt = nn.Parameter(torch.rand(1, 3, self.cfg.erp_h, self.cfg.erp_w), requires_grad=False)

    def render_face(self, rot: Tensor) -> Tensor:
        """Stub render: project gaussian means through rotation → 2D tint."""
        device = self.gaussians.device
        n = self.cfg.cube_face_res
        g = self.gaussians.view(n, n, 3)
        # simple view-dependent tint from rotation trace
        tint = (g @ rot.T).mean(dim=-1, keepdim=True).sigmoid()
        return tint.permute(2, 0, 1).unsqueeze(0).expand(1, 3, -1, -1)

    def render_erp(self) -> Tensor:
        h, w = self.cfg.erp_h, self.cfg.erp_w
        out = torch.zeros(1, 3, h, w, device=self.gaussians.device)
        n = min(32, h)
        for face in FACE_NAMES:
            dirs = face_grid(face, n, device=self.gaussians.device)
            th, ph = cartesian_to_spherical(dirs[:, 0], dirs[:, 1], dirs[:, 2])
            u, v = spherical_to_erp_uv(th, ph, h, w)
            ui = u.long().clamp(0, w - 1)
            vi = v.long().clamp(0, h - 1)
            tint = self.gaussians.mean().sigmoid().expand(3)
            out[0, :, vi, ui] = tint.unsqueeze(-1)
        return out.clamp(0, 1)

    def intra_loss(self) -> Tensor:
        views = view_rotations(include_transition=self.cfg.use_transition_plane, device=self.gaussians.device)
        losses: list[Tensor] = []
        gt_face = torch.rand(1, 3, self.cfg.cube_face_res, self.cfg.cube_face_res, device=self.gaussians.device)
        for _name, rot in views:
            pred = self.render_face(rot)
            losses.append(
                photometric_loss(
                    pred,
                    gt_face,
                    lambda_l1=self.cfg.lambda_l1,
                    lambda_dssim=self.cfg.lambda_dssim,
                )
            )
        return torch.stack(losses).mean()

    def inter_loss(self) -> Tensor:
        ec = self.render_erp()
        et = self.render_erp() * 0.95 + 0.05 * torch.randn_like(ec)
        er = stitch_erp(ec, et, shift_cols=self.cfg.padding_px if self.cfg.use_cube_padding else 0)
        return photometric_loss(
            er,
            self._erp_gt.detach(),
            lambda_l1=self.cfg.lambda_l1,
            lambda_dssim=self.cfg.lambda_dssim,
        )

    def forward(self) -> dict[str, Tensor]:
        loss_intra = self.intra_loss() if self.cfg.use_intra_inter else torch.tensor(0.0, device=self.gaussians.device)
        loss_inter = self.inter_loss() if self.cfg.use_intra_inter else loss_intra
        total = loss_intra + 0.5 * loss_inter
        erp = self.render_erp()
        return {
            "loss": total,
            "loss_intra": loss_intra.detach(),
            "loss_inter": loss_inter.detach(),
            "erp_rgb": erp.detach(),
            "padded_fov_rad": torch.tensor(
                padded_fov_rad(self.cfg.erp_h, self.cfg.padding_px if self.cfg.use_cube_padding else 0)
            ),
        }
