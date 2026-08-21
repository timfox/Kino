"""PanoWorld MLLM with SSCA (Fig. 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panoworld.config import PanoWorldConfig
from ltx_trainer.panoworld.metadata_graph import build_metadata_graph
from ltx_trainer.panoworld.ssca import SphericalSpatialCrossAttention
from ltx_trainer.panoworld.visual_encoder import ERPVisualEncoder


class PanoWorld(nn.Module):
    """Pano-native VLM stub: ERP encoder → SSCA → task heads."""

    def __init__(self, cfg: PanoWorldConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoWorldConfig()
        self.encoder = ERPVisualEncoder(self.cfg)
        self.ssca = SphericalSpatialCrossAttention(self.cfg)
        d = self.cfg.hidden_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.choice_head = nn.Linear(d, self.cfg.num_choices)
        self.bfov_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 4),
        )
        self.entity_head = nn.Linear(d, 8)

    def encode_panorama(self, rgb: Tensor) -> Tensor:
        h0 = self.encoder(rgb)
        return self.ssca(h0)

    def forward(
        self,
        rgb: Tensor,
        *,
        entity_semantics: Tensor | None = None,
        entity_bfov: Tensor | None = None,
        entity_depth: Tensor | None = None,
    ) -> dict[str, Tensor | object]:
        tokens = self.encode_panorama(rgb)
        pooled = self.pool(tokens.transpose(1, 2)).squeeze(-1)
        choice_logits = self.choice_head(pooled)
        bfov_pred = self.bfov_head(pooled)
        bfov_pred = torch.stack(
            [
                bfov_pred[..., 0],
                bfov_pred[..., 1],
                bfov_pred[..., 2].abs() + 1.0,
                bfov_pred[..., 3].abs() + 1.0,
            ],
            dim=-1,
        )
        out: dict[str, Tensor | object] = {
            "tokens": tokens,
            "choice_logits": choice_logits,
            "bfov_pred": bfov_pred,
        }
        if entity_semantics is not None and entity_bfov is not None and entity_depth is not None:
            ctx = self.entity_head(tokens.mean(dim=1))
            graphs = []
            n_ent = entity_semantics.shape[1]
            for bi in range(rgb.shape[0]):
                ctx_nodes = ctx[bi].unsqueeze(0).expand(n_ent, -1)
                graphs.append(
                    build_metadata_graph(
                        entity_semantics[bi],
                        entity_bfov[bi],
                        entity_depth[bi],
                        ctx_nodes,
                    )
                )
            out["metadata_graphs"] = graphs
        return out
