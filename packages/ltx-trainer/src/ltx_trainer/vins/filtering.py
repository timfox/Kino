"""VINS-120K multi-stage filtering pipeline (Sec. 3.2, Fig. 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class FilterScores:
    """Per-sample quality scores (higher = better)."""

    clarity: float
    luminance_ok: float
    saturation_ok: float
    texture: float
    instruction_clip: float
    preservation_l2: float
    aesthetic: float


def tenengrad_clarity(image: Tensor) -> float:
    """Tenengrad gradient magnitude proxy (Sec. 3.2)."""
    if image.dim() == 3:
        gray = image.mean(dim=0)
    else:
        gray = image
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    return float((gx.pow(2).mean() + gy.pow(2).mean()).sqrt().item())


def glcm_texture_score(image: Tensor) -> float:
    """Simplified texture richness via local variance."""
    if image.dim() == 3:
        gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        gray = image.unsqueeze(0).unsqueeze(0)
    pooled = torch.nn.functional.avg_pool2d(gray, 5, stride=1, padding=2)
    var = (gray - pooled).pow(2).mean()
    return float(var.item())


def luminance_score(image: Tensor, *, low: float = 0.1, high: float = 0.9) -> float:
    lum = float(image.mean().item())
    return 1.0 if low <= lum <= high else 0.0


def saturation_score(image: Tensor, *, max_sat: float = 0.85) -> float:
    if image.shape[0] < 3:
        return 1.0
    maxc, _ = image.max(dim=0)
    minc, _ = image.min(dim=0)
    sat = ((maxc - minc) / maxc.clamp(min=1e-6)).mean()
    return 1.0 if float(sat.item()) <= max_sat else 0.0


def instruction_following_score(
    edited: Tensor,
    original: Tensor,
    *,
    clip_sim: float | None = None,
    preserve_l2: float | None = None,
) -> float:
    """Joint edited-region CLIP + non-edited L2 proxy (Sec. 3.2)."""
    if clip_sim is not None and preserve_l2 is not None:
        return 0.6 * clip_sim + 0.4 * max(0.0, 1.0 - preserve_l2)
    diff = (edited - original).pow(2).mean()
    return float(max(0.0, 1.0 - diff.item()))


def compute_filter_scores(
    original: Tensor,
    edited: Tensor,
    *,
    clip_sim: float | None = None,
) -> FilterScores:
    """Aggregate filtering stage scores for one triplet."""
    preserve = float((original - edited).pow(2).mean().item())
    return FilterScores(
        clarity=tenengrad_clarity(edited),
        luminance_ok=luminance_score(edited),
        saturation_ok=saturation_score(edited),
        texture=glcm_texture_score(edited),
        instruction_clip=instruction_following_score(edited, original, clip_sim=clip_sim, preserve_l2=preserve),
        preservation_l2=1.0 - min(1.0, preserve),
        aesthetic=0.5 * luminance_score(edited) + 0.5 * saturation_score(edited),
    )


def passes_filter(
    scores: FilterScores,
    *,
    clarity_threshold: float = 0.01,
    min_instruction: float = 0.3,
) -> bool:
    return (
        scores.clarity >= clarity_threshold
        and scores.luminance_ok > 0
        and scores.saturation_ok > 0
        and scores.instruction_clip >= min_instruction
    )


def retain_top_fraction(
    batch_scores: list[FilterScores],
    *,
    fraction: float = 0.2,
) -> list[int]:
    """Keep top fraction by composite score (Sec. 3.2, 20% retained)."""
    composite = [
        s.clarity + s.texture + s.instruction_clip + s.aesthetic + s.preservation_l2
        for s in batch_scores
    ]
    k = max(1, int(len(batch_scores) * fraction))
    order = sorted(range(len(batch_scores)), key=lambda i: composite[i], reverse=True)
    return order[:k]
