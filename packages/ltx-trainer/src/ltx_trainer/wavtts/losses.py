"""FM x-prediction and multi-scale mel auxiliary loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def flow_matching_x_loss(
    x_pred: Tensor,
    x1: Tensor,
    mask: Tensor,
    t: Tensor,
    *,
    t_clip_max: float = 0.98,
) -> Tensor:
    """Eq. (3): MSE of (xθ - x1) ⊙ m / (1-t), with t clipped."""
    t_safe = t.clamp(max=t_clip_max).view(-1, 1, 1)
    diff = (x_pred - x1) * mask
    return (diff / (1.0 - t_safe)).pow(2).mean()


def multi_scale_mel_stub(x_pred: Tensor, x1: Tensor, mask: Tensor) -> Tensor:
    """Lightweight mel proxy: STFT magnitude L1 on masked regions (Eq. 6 stub)."""
    def _mag(w: Tensor) -> Tensor:
        spec = torch.stft(
            w,
            n_fft=256,
            hop_length=64,
            win_length=256,
            return_complex=True,
        )
        return spec.abs().log1p()

    m = mask.squeeze(-1) if mask.dim() == 3 else mask
    loss = 0.0
    for i in range(x_pred.shape[0]):
        if m[i].sum() < 1:
            continue
        loss = loss + F.l1_loss(_mag(x_pred[i]), _mag(x1[i]))
    n = max(int(mask.shape[0]), 1)
    return torch.as_tensor(loss / n, dtype=x_pred.dtype, device=x_pred.device)
