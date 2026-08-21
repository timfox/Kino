"""Disentangled camera style extraction (Sec. 3.3, Eq. 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def normalized_cross_correlation(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """LNCC trajectory loss term (higher is better correlation → use 1 - NCC as loss)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch {pred.shape} vs {target.shape}")
    p = pred - pred.mean()
    t = target - target.mean()
    num = (p * t).sum()
    den = torch.sqrt((p * p).sum() * (t * t).sum() + eps)
    return num / den


def ncc_loss(pred: Tensor, target: Tensor) -> Tensor:
    return 1.0 - normalized_cross_correlation(pred, target)


def info_nce_loss(anchor: Tensor, positive: Tensor, negatives: Tensor, temperature: float = 0.07) -> Tensor:
    """Contrastive loss: anchor matches positive over negatives along batch dim."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negatives, dim=-1)
    pos_logit = (a * p).sum(dim=-1, keepdim=True) / temperature
    neg_logits = (a.unsqueeze(1) * n).sum(dim=-1) / temperature
    logits = torch.cat([pos_logit, neg_logits], dim=-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def mutual_information_penalty(z_content: Tensor, z_style: Tensor) -> Tensor:
    """L_MI: squared cosine similarity between content and style (Eq. 4)."""
    c = F.normalize(z_content, dim=-1)
    s = F.normalize(z_style, dim=-1)
    return (c * s).sum(dim=-1).pow(2).mean()


class StyleTrajectoryHead(nn.Module):
    """Map per-frame style embedding to intrinsic trajectory (Table 3 regression)."""

    def __init__(self, style_dim: int, num_params: int) -> None:
        super().__init__()
        self.linear = nn.Linear(style_dim, num_params)

    def forward(self, z_style: Tensor) -> Tensor:
        return self.linear(z_style)


class DisentangledStyleEmbedder(nn.Module):
    """Lightweight dual-branch embedder for synthetic triplet training (Fig. 6)."""

    def __init__(self, in_dim: int, style_dim: int, content_dim: int) -> None:
        super().__init__()
        self.content = nn.Sequential(nn.Linear(in_dim, content_dim), nn.SiLU(), nn.Linear(content_dim, content_dim))
        self.style = nn.Sequential(nn.Linear(in_dim, style_dim), nn.SiLU(), nn.Linear(style_dim, style_dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.content(x), self.style(x)


def style_extraction_loss(
    tau_pred: Tensor,
    tau_gt: Tensor,
    z_a_c: Tensor,
    z_c_c: Tensor,
    z_a_s: Tensor,
    z_s_s: Tensor,
    *,
    lambda_tau: float = 1.0,
    lambda_c: float = 1.0,
    lambda_s: float = 1.0,
    lambda_mi: float = 0.1,
    temperature: float = 0.07,
) -> dict[str, Tensor]:
    """Eq. (4) with batch negatives = other samples in batch."""
    l_tau = ncc_loss(tau_pred, tau_gt)
    # content: anchor vs same-scene
    l_c = info_nce_loss(z_a_c, z_c_c, z_c_c.roll(1, 0), temperature=temperature)
    # style: anchor vs same-style-different-scene
    l_s = info_nce_loss(z_a_s, z_s_s, z_s_s.roll(1, 0), temperature=temperature)
    l_mi = mutual_information_penalty(z_a_c, z_a_s)
    total = lambda_tau * l_tau + lambda_c * l_c + lambda_s * l_s + lambda_mi * l_mi
    return {"total": total, "tau": l_tau, "content": l_c, "style": l_s, "mi": l_mi}
