"""Human Anatomical Preference (HAP) pair schema and curation gates (Sec. 4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ltx_trainer.asap.config import ASAPConfig
from ltx_trainer.asap.degradation import background_mse, region_mse


@dataclass
class HAPPair:
    """One preference tuple (x_w, x_l, c) with optional anatomy mask."""

    caption: str
    positive: Tensor
    negative: Tensor
    mask: Tensor | None = None
    degradation: str = "image_space"


def pair_passes_hap_filters(
    positive: Tensor,
    negative: Tensor,
    mask: Tensor,
    *,
    cfg: ASAPConfig | None = None,
) -> bool:
    """Heuristic curation: visible region change + stable background (Sec. 4.1).

    Uses MSE proxies instead of full SSIM/LPIPS so tests run without extra deps.
    Paper targets: region SSIM≈0.53, bg SSIM≈0.98.
    """
    cfg = cfg or ASAPConfig()
    reg = region_mse(positive, negative, mask)
    bg = background_mse(positive, negative, mask)
    # Map MSE thresholds to paper-inspired gates (tunable)
    region_ok = reg >= cfg.region_lpips_min * 0.05
    bg_ok = bg <= cfg.background_lpips_max * 2.0
    return region_ok and bg_ok


def hap_batch_collate(pairs: list[HAPPair]) -> dict[str, Tensor | list[str]]:
    """Stack tensors for training loop (same spatial size required)."""
    if not pairs:
        raise ValueError("empty pair list")
    return {
        "caption": [p.caption for p in pairs],
        "positive": torch.stack([p.positive for p in pairs]),
        "negative": torch.stack([p.negative for p in pairs]),
        "mask": torch.stack([p.mask for p in pairs if p.mask is not None]),
    }
