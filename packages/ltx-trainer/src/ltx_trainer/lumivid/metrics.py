"""Evaluation metrics (Sec. 4, Tables 1–5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.hdr_ingest import pu21_forward_linear_abs


def pu21_psnr(pred: Tensor, gt: Tensor, *, l_peak: float = 4000.0) -> float:
    """PU21-PSNR proxy on scene-linear HDR."""
    p = _to_pu21(pred, l_peak)
    g = _to_pu21(gt, l_peak)
    mse = F.mse_loss(p, g).item()
    if mse <= 0:
        return 99.0
    return float(10.0 * torch.log10(torch.tensor(1.0 / mse)))


def _to_pu21(hdr: Tensor, l_peak: float) -> Tensor:
    x = hdr.clamp(min=0.0)
    peak = x.amax().clamp(min=1e-10)
    return pu21_forward_linear_abs(x * (l_peak / peak))


def roundtrip_error(scene: Tensor, roundtrip: Tensor) -> float:
    return float(F.l1_loss(roundtrip.clamp(min=0.0), scene.clamp(min=0.0)))


def temporal_flicker(video_cfhw: Tensor) -> float:
    """Normalized std of per-frame mean luminance (Table 3)."""
    c, f, _, _ = video_cfhw.shape
    y = 0.2126 * video_cfhw[0] + 0.7152 * video_cfhw[1] + 0.0722 * video_cfhw[2]
    means = y.reshape(f, -1).mean(dim=1)
    return float(means.std() / means.mean().clamp(min=1e-6))


def f2f_psnr(video_cfhw: Tensor) -> float:
    """Frame-to-frame PSNR on luminance."""
    c, f, h, w = video_cfhw.shape
    if f < 2:
        return 99.0
    y = 0.2126 * video_cfhw[0] + 0.7152 * video_cfhw[1] + 0.0722 * video_cfhw[2]
    psnrs = []
    for i in range(f - 1):
        mse = F.mse_loss(y[i], y[i + 1]).item()
        if mse <= 0:
            psnrs.append(99.0)
        else:
            psnrs.append(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
    return float(sum(psnrs) / len(psnrs))
