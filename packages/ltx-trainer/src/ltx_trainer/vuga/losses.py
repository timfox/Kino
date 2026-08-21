"""MSE quality regression loss (Eq. 13)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mse_quality_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target)


def plcc_srcc_stub(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Pearson / Spearman correlation for smoke metrics."""
    p = pred.detach().float().flatten()
    t = target.detach().float().flatten()
    if p.numel() < 2:
        return {"plcc": 0.0, "srcc": 0.0}
    p_c = p - p.mean()
    t_c = t - t.mean()
    plcc = float((p_c * t_c).sum() / (p_c.norm() * t_c.norm() + 1e-8))
    rp = p.argsort().argsort().float()
    rt = t.argsort().argsort().float()
    rp_c = rp - rp.mean()
    rt_c = rt - rt.mean()
    srcc = float((rp_c * rt_c).sum() / (rp_c.norm() * rt_c.norm() + 1e-8))
    return {"plcc": plcc, "srcc": srcc}
