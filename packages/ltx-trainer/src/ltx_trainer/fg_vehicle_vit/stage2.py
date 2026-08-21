"""Stage 2 ViT classifier, focal loss, and class weights (§2.5)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig


def inverse_frequency_class_weights(counts: list[int]) -> Tensor:
    """Eq. (1): w_c = N / (n_c * C), normalized so sum(w) = C."""
    n = sum(counts)
    c = len(counts)
    raw = [n / (nc * c) if nc > 0 else 1.0 for nc in counts]
    s = sum(raw)
    scale = c / s if s > 0 else 1.0
    return torch.tensor([r * scale for r in raw], dtype=torch.float32)


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 2.0,
    weight: Tensor | None = None,
) -> Tensor:
    """Focal loss (Lin et al., 2017) with optional per-class weights."""
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


class ViTClassifierHead(nn.Module):
    """Minimal ViT-style patch encoder + linear head for smoke tests."""

    def __init__(self, cfg: FgVehicleVitConfig | None = None, embed_dim: int = 128) -> None:
        super().__init__()
        cfg = cfg or FgVehicleVitConfig()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 3, 224, 224)
        h = self.patch(x).flatten(2).transpose(1, 2)  # (B, tokens, D)
        h = self.norm(h.mean(dim=1))
        return self.head(h)


def training_step_demo(cfg: FgVehicleVitConfig | None = None) -> dict[str, float]:
    cfg = cfg or FgVehicleVitConfig()
    torch.manual_seed(51)
    counts = [10375, 3888, 1630, 510, 92, 86]  # Table 1
    weights = inverse_frequency_class_weights(counts)
    model = ViTClassifierHead(cfg)
    x = torch.randn(8, 3, cfg.crop_size, cfg.crop_size)
    y = torch.randint(0, cfg.num_classes, (8,))
    logits = model(x)
    loss = focal_loss(logits, y, gamma=cfg.focal_gamma, weight=weights)
    return {
        "focal_loss": float(loss.detach()),
        "weight_sum": float(weights.sum()),
        "logits_shape": float(logits.shape[-1]),
    }
