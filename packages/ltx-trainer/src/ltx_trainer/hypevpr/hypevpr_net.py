"""HypeVPR network stub (Sec. 4.1–4.5)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.hypevpr.aggregation import HierarchicalAggregationModule, LevelAggregator
from ltx_trainer.hypevpr.config import HypeVPRConfig
from ltx_trainer.hypevpr.hierarchy import slice_panorama_windows
from ltx_trainer.hypevpr.poincare import expmap0


class HypeVPRStub(nn.Module):
    """Shared backbone + query path F_q and database path F_d with HAM."""

    def __init__(self, cfg: HypeVPRConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HypeVPRConfig()
        p = 8
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=p, stride=p),
            nn.ReLU(inplace=True),
        )
        self.query_agg = LevelAggregator(32, self.cfg.descriptor_dim, self.cfg.gem_p)
        self.ham = HierarchicalAggregationModule(self.cfg, backbone_dim=32)

    def encode_query(self, query: Tensor) -> tuple[Tensor, Tensor]:
        """I_q → d_q (Euclidean), h_q (hyperbolic) (Eq. 7–9)."""
        fq = self.backbone(query)
        dq = self.query_agg(fq)
        hq = expmap0(dq, c=self.cfg.curvature)
        return dq, hq

    def encode_database(self, pano: Tensor) -> dict[int, list[Tensor]]:
        """I_d → hierarchical hyperbolic descriptors H_d (Eq. 10–15)."""
        windows = slice_panorama_windows(pano, self.cfg.hierarchy_levels)
        feats = [self.backbone(w) for w in windows]
        return self.ham(feats)

    def top_descriptor(self, tree: dict[int, list[Tensor]]) -> Tensor:
        return tree[1][0]

    def leaf_descriptors(self, tree: dict[int, list[Tensor]]) -> list[Tensor]:
        return tree[max(tree.keys())]

    def forward(
        self,
        query: Tensor,
        pano: Tensor,
    ) -> dict[str, Tensor | dict[int, list[Tensor]]]:
        dq, hq = self.encode_query(query)
        tree = self.encode_database(pano)
        return {"dq": dq, "hq": hq, "db_tree": tree, "h_top": self.top_descriptor(tree)}
