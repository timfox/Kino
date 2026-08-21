"""VINS-4KEval metrics: ImageJudge dims, VIEScore, patch-FID proxy."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def imagejudge_scores(
    *,
    instruction_adherence: float = 4.0,
    editing_quality: float = 4.0,
    detail_preservation: float = 4.0,
) -> dict[str, float]:
    """ImageJudge three-axis summary (Sec. 5)."""
    avg = (instruction_adherence + editing_quality + detail_preservation) / 3.0
    return {
        "instruction_adherence": instruction_adherence,
        "editing_quality": editing_quality,
        "detail_preservation": detail_preservation,
        "average": avg,
    }


def viescore(
    semantic_consistency: float,
    perceptual_quality: float,
) -> dict[str, float]:
    """Overall = sqrt(SC × PQ) (Sec. 5)."""
    import math

    overall = math.sqrt(max(0.0, semantic_consistency) * max(0.0, perceptual_quality))
    return {
        "semantic_consistency": semantic_consistency,
        "perceptual_quality": perceptual_quality,
        "overall": overall,
    }


def patch_fid_proxy(
    real: Tensor,
    fake: Tensor,
    *,
    patch: int = 64,
    stride: int = 32,
) -> float:
    """Patch-level feature distance proxy for pFID (lower is better)."""
    if real.dim() == 3:
        real = real.mean(dim=0, keepdim=True)
        fake = fake.mean(dim=0, keepdim=True)
    c, h, w = real.shape
    if h < patch or w < patch:
        return float((real - fake).pow(2).mean().item())
    feats_r, feats_f = [], []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            pr = real[:, y : y + patch, x : x + patch]
            pf = fake[:, y : y + patch, x : x + patch]
            feats_r.append(pr.reshape(-1))
            feats_f.append(pf.reshape(-1))
    fr = torch.stack(feats_r).float()
    ff = torch.stack(feats_f).float()
    mu_r, mu_f = fr.mean(0), ff.mean(0)
    cov_r = torch.cov(fr.T) if fr.shape[0] > 1 else torch.eye(fr.shape[1])
    cov_f = torch.cov(ff.T) if ff.shape[0] > 1 else torch.eye(ff.shape[1])
    diff = mu_r - mu_f
    # simplified Fréchet distance
    return float((diff.pow(2).sum() + (cov_r - cov_f).pow(2).mean()).sqrt().item())


def high_frequency_energy(image: Tensor) -> float:
    """High-frequency energy ratio for detail fidelity."""
    if image.dim() == 3:
        gray = image.mean(dim=0)
    else:
        gray = image
    spec = torch.fft.fft2(gray, norm="ortho")
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    mask = torch.ones(h, w, device=spec.device, dtype=torch.bool)
    mask[cy - h // 8 : cy + h // 8, cx - w // 8 : cx + w // 8] = False
    hf = spec[mask].abs().pow(2).mean()
    total = spec.abs().pow(2).mean().clamp(min=1e-8)
    return float((hf / total).item())
