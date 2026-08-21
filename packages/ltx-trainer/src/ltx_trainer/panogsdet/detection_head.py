"""Gaussian-guided 3D detection head (Sec. II-C, TR3D-style)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.panogsdet.config import PanoGSDetConfig
from ltx_trainer.panogsdet.gaussian_lift import SemanticGaussianState


class GaussianGuidedDetectionHead(nn.Module):
    """
    Foreground Gaussians → 3D box center offset, size, yaw, confidence.
  Class label initialized from Gaussian argmax category.
    """

    def __init__(self, cfg: PanoGSDetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PanoGSDetConfig()
        d = self.cfg.feature_dim
        hidden = max(d, 64)
        self.backbone = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Linear(hidden, 7)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(
        self,
        state: SemanticGaussianState,
        *,
        max_proposals: int = 512,
    ) -> dict[str, Tensor]:
        logits = state.category_logits
        probs = torch.softmax(logits, dim=-1)
        cls_id = probs.argmax(dim=-1)
        fg_mask = torch.zeros_like(cls_id, dtype=torch.bool)
        for cid in self.cfg.foreground_class_ids:
            fg_mask |= cls_id == cid
        b = state.centers.shape[0]
        boxes = []
        scores = []
        labels = []
        for bi in range(b):
            idx = torch.where(fg_mask[bi])[0]
            if idx.numel() == 0:
                boxes.append(torch.zeros(0, 7, device=state.centers.device))
                scores.append(torch.zeros(0, device=state.centers.device))
                labels.append(torch.zeros(0, dtype=torch.long, device=state.centers.device))
                continue
            if idx.numel() > max_proposals:
                idx = idx[:max_proposals]
            feat = state.features[bi, idx]
            h = self.backbone(feat)
            raw = self.box_head(h)
            offset = raw[:, :3]
            size = torch.nn.functional.softplus(raw[:, 3:6]) + 1e-3
            yaw = raw[:, 6:7]
            center = state.centers[bi, idx] + offset
            conf = torch.sigmoid(self.conf_head(h).squeeze(-1))
            box = torch.cat([center, size, yaw], dim=-1)
            boxes.append(box)
            scores.append(conf)
            labels.append(cls_id[bi, idx])
        return {"boxes": boxes, "scores": scores, "labels": labels, "class_probs": probs}
