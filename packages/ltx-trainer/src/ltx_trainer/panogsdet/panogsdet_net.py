"""Full PanoGSDet network (Fig. 2)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig
from ltx_trainer.panogsdet.depth_branch import PanoramicDepthBranch
from ltx_trainer.panogsdet.detection_head import GaussianGuidedDetectionHead
from ltx_trainer.panogsdet.gaussian_lift import SemanticGaussianLifting
from ltx_trainer.panogsdet.gaussian_optimize import SemanticGaussianOptimizer


class PanoGSDet(nn.Module):
    def __init__(self, cfg: PanoGSDetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoGSDetConfig()
        self.depth_branch = PanoramicDepthBranch(self.cfg)
        self.lifting = SemanticGaussianLifting(self.cfg)
        self.optimizer = SemanticGaussianOptimizer(self.cfg)
        self.det_head = GaussianGuidedDetectionHead(self.cfg)

    def forward(self, rgb: Tensor, *, face_size: int = 32, max_gaussians: int = 1024) -> dict[str, Any]:
        depth, sem = self.depth_branch(rgb)
        state = self.lifting(depth, sem, max_points=max_gaussians)
        state_opt = self.optimizer(state)
        det = self.det_head(state_opt, max_proposals=max_gaussians)
        from ltx_trainer.panogsdet.cubemap import render_semantic_cubemap

        rendered = render_semantic_cubemap(state_opt, face_size=face_size, max_points=max_gaussians)
        return {
            "depth": depth,
            "sem_feat": sem,
            "gaussians": state_opt,
            "detection": det,
            "cubemap": rendered,
        }
