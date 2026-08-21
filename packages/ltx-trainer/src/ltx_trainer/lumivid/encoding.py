"""HDR encodings for VAE manifold alignment (Sec. 3.1, Fig. 4)."""

from __future__ import annotations

import math
from enum import Enum

import torch
from ltx_core.hdr import LogC3
from torch import Tensor

from ltx_trainer.hdr_ingest import linear_scene_to_logc3_display, linear_scene_to_pu21_display, pu21_inverse_to_linear_abs


class HdREncoding(str, Enum):
    LOGC3 = "logc3"
    PQ = "pq"
    HLG = "hlg"
    ACES = "aces"


_LOGC3 = LogC3()


def compress_scene_linear(scene: Tensor, encoding: HdREncoding | str = HdREncoding.LOGC3, *, l_peak: float = 4000.0) -> Tensor:
    """Map scene-linear ``[0,∞)`` RGB to bounded display codes ``[0,1]``."""
    enc = HdREncoding(encoding)
    x = scene.clamp(min=0.0)
    if enc == HdREncoding.LOGC3:
        return linear_scene_to_logc3_display(x)
    if enc == HdREncoding.PQ:
        return linear_scene_to_pu21_display(x, l_peak=l_peak)
    if enc == HdREncoding.HLG:
        return _hlg_oetf(x)
    return _aces_filmic(x)


def decompress_to_scene_linear(codes: Tensor, encoding: HdREncoding | str = HdREncoding.LOGC3, *, l_peak: float = 4000.0) -> Tensor:
    enc = HdREncoding(encoding)
    y = codes.clamp(0.0, 1.0)
    if enc == HdREncoding.LOGC3:
        return _LOGC3.decompress(y)
    if enc == HdREncoding.PQ:
        peak = y.amax().clamp(min=1e-10)
        return pu21_inverse_to_linear_abs(y) * (peak / l_peak)
    if enc == HdREncoding.HLG:
        return _hlg_inverse(y)
    return _aces_inverse(y)


def vae_input_from_scene_linear(scene: Tensor, encoding: HdREncoding | str = HdREncoding.LOGC3, *, l_peak: float = 4000.0) -> Tensor:
    """LogC3/PQ/etc. codes → VAE native ``[-1,1]``."""
    y = compress_scene_linear(scene, encoding, l_peak=l_peak)
    return y * 2.0 - 1.0


def vae_output_to_scene_linear(vae_rgb: Tensor, encoding: HdREncoding | str = HdREncoding.LOGC3, *, l_peak: float = 4000.0) -> Tensor:
    """VAE output ``[0,1]`` or ``[-1,1]`` → scene-linear."""
    if vae_rgb.min() < 0.0:
        y = ((vae_rgb + 1.0) * 0.5).clamp(0.0, 1.0)
    else:
        y = vae_rgb.clamp(0.0, 1.0)
    return decompress_to_scene_linear(y, encoding, l_peak=l_peak)


def _hlg_oetf(x: Tensor) -> Tensor:
    a, b = 0.17883277, 0.28466892
    c, d = 0.55991073, 0.07270922
    xc = x.clamp(min=0.0)
    low = (c * xc) ** 0.5
    high = a * torch.log(xc + b) + d
    return torch.where(xc <= 1.0 / 12.0, low, high).clamp(0.0, 1.0)


def _hlg_inverse(y: Tensor) -> Tensor:
    a, b = 0.17883277, 0.28466892
    c, d = 0.55991073, 0.07270922
    yc = y.clamp(0.0, 1.0)
    low = (yc / c) ** 2
    high = torch.exp((yc - d) / a) - b
    return torch.where(yc <= 0.5, low, high).clamp(min=0.0)


def _aces_filmic(x: Tensor) -> Tensor:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    xc = x.clamp(min=0.0)
    num = xc * (a * xc + b)
    den = xc * (c * xc + d) + e
    return (num / den.clamp(min=1e-8)).clamp(0.0, 1.0)


def _aces_inverse(y: Tensor) -> Tensor:
    """Approximate inverse of ACES filmic for roundtrip tests."""
    yc = y.clamp(0.0, 1.0)
    return (yc / (1.03 - yc)).clamp(min=0.0)


def kl_histogram(p: Tensor, q: Tensor, *, bins: int = 64) -> float:
    """KL(p||q) from flattened tensor histograms (Sec. 3.1 analysis)."""
    pf = p.detach().reshape(-1).float().cpu()
    qf = q.detach().reshape(-1).float().cpu()
    lo = min(float(pf.min()), float(qf.min()))
    hi = max(float(pf.max()), float(qf.max()))
    if hi - lo < 1e-8:
        return 0.0
    p_hist = torch.histc(pf, bins=bins, min=lo, max=hi) + 1e-8
    q_hist = torch.histc(qf, bins=bins, min=lo, max=hi) + 1e-8
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float((p_hist * (p_hist / q_hist).log()).sum())
