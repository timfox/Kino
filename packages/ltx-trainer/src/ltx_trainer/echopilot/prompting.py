"""Scale-space semantic prompting and VFM auxiliary points (Sec. 2.2)."""

from __future__ import annotations

import torch
from torch import Tensor


def crop_boxes(
    height: int,
    width: int,
    center: tuple[float, float],
    scale_factors: tuple[float, ...],
) -> list[tuple[int, int, int, int]]:
    """Center crops as (x0, y0, x1, y1) for each scale factor σ."""
    cx, cy = center
    boxes: list[tuple[int, int, int, int]] = []
    for sigma in scale_factors:
        half_w = max(4, int(width * sigma / 2))
        half_h = max(4, int(height * sigma / 2))
        x0 = max(0, int(cx - half_w))
        y0 = max(0, int(cy - half_h))
        x1 = min(width, int(cx + half_w))
        y1 = min(height, int(cy + half_h))
        boxes.append((x0, y0, x1, y1))
    return boxes


def vfm_cosine_map(features: Tensor, anchor: tuple[int, int]) -> Tensor:
    """Eq. (5): dense cosine similarity to anchor feature."""
    if features.dim() != 3:
        raise ValueError("features must be (H, W, D)")
    h, w, _ = features.shape
    ay = max(0, min(h - 1, anchor[0]))
    ax = max(0, min(w - 1, anchor[1]))
    anchor_feat = features[ay, ax]
    anchor_feat = anchor_feat / anchor_feat.norm().clamp(min=1e-8)
    flat = features.reshape(-1, features.shape[-1])
    flat = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = (flat @ anchor_feat).reshape(h, w)
    return sim.clamp(-1.0, 1.0)


def nms_peaks(
    similarity: Tensor,
    box: tuple[int, int, int, int],
    *,
    max_points: int = 3,
    radius: int = 6,
) -> list[tuple[int, int]]:
    """Local maxima inside box via non-maximum suppression."""
    x0, y0, x1, y1 = box
    region = similarity[y0:y1, x0:x1].clone()
    if region.numel() == 0:
        return []
    peaks: list[tuple[int, int]] = []
    for _ in range(max_points):
        flat_idx = int(region.reshape(-1).argmax().item())
        if region.reshape(-1)[flat_idx] <= 0:
            break
        rh, rw = region.shape
        py, px = divmod(flat_idx, rw)
        gy, gx = y0 + py, x0 + px
        peaks.append((gy, gx))
        y_lo = max(0, py - radius)
        y_hi = min(rh, py + radius + 1)
        x_lo = max(0, px - radius)
        x_hi = min(rw, px + radius + 1)
        region[y_lo:y_hi, x_lo:x_hi] = -1.0
    return peaks


def synthesize_prompts(
    similarity: Tensor,
    user_point: tuple[int, int],
    box: tuple[int, int, int, int],
    *,
    max_aux: int = 3,
    nms_radius: int = 6,
) -> list[tuple[int, int]]:
    """Return {p0, p1, p2, p3} with up to max_aux auxiliary positives."""
    aux = nms_peaks(similarity, box, max_points=max_aux, radius=nms_radius)
    prompts = [user_point]
    for p in aux:
        if p != user_point and p not in prompts:
            prompts.append(p)
        if len(prompts) >= 1 + max_aux:
            break
    return prompts[: 1 + max_aux]


def stub_attribution_from_similarity(similarity: Tensor, box: tuple[int, int, int, int]) -> Tensor:
    """Coarse attribution proxy when VLM gradients are unavailable."""
    x0, y0, x1, y1 = box
    attr = torch.zeros_like(similarity)
    attr[y0:y1, x0:x1] = similarity[y0:y1, x0:x1].relu()
    return attr


def random_vfm_features(height: int, width: int, dim: int, *, device: torch.device, seed: int) -> Tensor:
    """Synthetic dense features for smoke tests."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    feat = torch.randn(height, width, dim, generator=gen, device=device)
    return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
