"""Codec-stream tokenization: GOP partition + block saliency (Sec. 2.2, Eq. 2–6)."""

from __future__ import annotations

import torch
from torch import Tensor


def bit_cost_per_bin(p_frame_bytes: Tensor, num_bins: int) -> Tensor:
    """Aggregate P-frame packet bytes into temporal bins (Eq. 2)."""
    if p_frame_bytes.numel() == 0:
        return torch.zeros(num_bins)
    edges = torch.linspace(0, p_frame_bytes.shape[0], num_bins + 1).long()
    costs = []
    for b in range(num_bins):
        seg = p_frame_bytes[edges[b] : edges[b + 1]]
        costs.append(seg.sum() if seg.numel() else torch.tensor(0.0))
    return torch.stack(costs)


def adaptive_gop_boundaries(
    bin_costs: Tensor,
    *,
    k_tar: int,
    l_min: int,
    l_max: int,
) -> list[tuple[int, int]]:
    """Eq. (2–4): bit-cost quota triggers + local valley refinement."""
    b = bin_costs.numel()
    if b == 0:
        return [(0, 0)]
    theta = bin_costs.sum() / max(1, k_tar)
    groups: list[tuple[int, int]] = []
    sk = 0
    while sk < b:
        acc = 0.0
        ik = sk
        for i in range(sk, min(b, sk + l_max)):
            acc += float(bin_costs[i])
            span = i - sk + 1
            if span >= l_max or (span >= l_min and acc >= theta):
                ik = i
                break
            ik = i
        window = range(max(sk + l_min - 1, sk), min(b, ik + 2))
        if window:
            ck = min(window, key=lambda j: (float(bin_costs[j]), abs(j - ik)))
        else:
            ck = ik
        groups.append((sk, ck))
        sk = ck + 1
    return groups


def motion_residual_saliency(
    motion_mag: Tensor,
    residual: Tensor,
) -> Tensor:
    """St(x) = M^t(x) + R^t(x) with percentile normalization (Sec. 2.2)."""
    def norm(x: Tensor) -> Tensor:
        p = torch.quantile(x.flatten().float(), 0.95).clamp(min=1e-6)
        return (x / p).clamp(0.0, 1.0)

    return norm(motion_mag) + norm(residual.abs() - 128.0)


def block_scores_from_saliency(saliency: Tensor, block_p: int = 16) -> Tensor:
    """2×2 patch-block aggregation (Sec. 2.2)."""
    h, w = saliency.shape
    bh, bw = h // (2 * block_p), w // (2 * block_p)
    if bh == 0 or bw == 0:
        return saliency.new_zeros(1, 1)
    blocks = []
    for i in range(bh):
        for j in range(bw):
            patch = saliency[
                2 * i * block_p : 2 * (i + 1) * block_p,
                2 * j * block_p : 2 * (j + 1) * block_p,
            ]
            blocks.append(patch.sum())
    return torch.stack(blocks).view(bh, bw)


def stratified_frame_weight(
    rank_in_frame: int,
    score: float,
    *,
    lam: float = 0.5,
    alpha_peak: float = 0.1,
) -> float:
    """Eq. (5): attenuate repeated blocks from same frame."""
    return score / (1.0 + lam * rank_in_frame) ** 0.5 + alpha_peak * score
