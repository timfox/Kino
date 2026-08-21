"""Confidence-based abstention at Stage 2 (§2.6)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig

UNKNOWN_LABEL = "unknown"


def classify_with_abstention(
    logits: Tensor,
    cfg: FgVehicleVitConfig | None = None,
) -> tuple[str, float, bool]:
    """Return (label, confidence, abstained)."""
    cfg = cfg or FgVehicleVitConfig()
    probs = torch.softmax(logits, dim=-1)
    if logits.dim() == 1:
        probs = probs.unsqueeze(0)
        logits = logits.unsqueeze(0)
    conf, idx = probs.max(dim=-1)
    conf_v = float(conf.item())
    label = cfg.fine_grained_classes[int(idx.item())]
    if conf_v < cfg.abstention_threshold:
        return UNKNOWN_LABEL, conf_v, True
    return label, conf_v, False


def abstention_rate(num_abstained: int, num_total: int) -> float:
    if num_total <= 0:
        return 0.0
    return num_abstained / num_total
