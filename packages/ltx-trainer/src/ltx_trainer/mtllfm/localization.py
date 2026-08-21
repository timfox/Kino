"""Post-hoc temporal localization (Sec. 3.5, Eq. 8–9)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sharpen_attention(alpha: Tensor, *, temperature: float = 0.5) -> Tensor:
    """Eq. (8): temperature sharpening of attention."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    sharp = alpha.pow(1.0 / temperature)
    return sharp / sharp.sum()


def align_modalities(
    alpha_a: Tensor,
    alpha_v: Tensor,
    *,
    num_bins: int,
    w_a: float,
    w_v: float,
) -> Tensor:
    """Align audio/visual attention to ``N`` bins and combine (Eq. 9)."""
    # audio: max-pool to bins; visual: interpolate
    a = alpha_a.unsqueeze(0).unsqueeze(0)
    v = alpha_v.unsqueeze(0).unsqueeze(0)
    a_bins = F.adaptive_max_pool1d(a, num_bins).squeeze()
    v_bins = F.interpolate(v, size=num_bins, mode="linear", align_corners=False).squeeze()
    return w_a * a_bins + w_v * v_bins


def localize_peak_expansion(
    beta: Tensor,
    *,
    segment_seconds: float = 5.0,
) -> tuple[float, float]:
    """
    Peak bin + expansion while attention exceeds mean → continuous interval.

    Returns:
        (start_sec, end_sec)
    """
    n = beta.numel()
    mean = beta.mean()
    peak = int(beta.argmax().item())
    left, right = peak, peak
    while left > 0 and beta[left - 1] >= mean:
        left -= 1
    while right < n - 1 and beta[right + 1] >= mean:
        right += 1
    bin_dur = segment_seconds / n
    return left * bin_dur, (right + 1) * bin_dur


def localize_laughter(
    alpha_a: Tensor,
    alpha_v: Tensor,
    *,
    w_a: float,
    w_v: float,
    num_bins: int = 10,
    temperature: float = 0.5,
    segment_seconds: float = 5.0,
) -> dict[str, float]:
    """Full localization pipeline from modality attentions."""
    a_sharp = sharpen_attention(alpha_a, temperature=temperature)
    v_sharp = sharpen_attention(alpha_v, temperature=temperature)
    beta = align_modalities(a_sharp, v_sharp, num_bins=num_bins, w_a=w_a, w_v=w_v)
    start, end = localize_peak_expansion(beta, segment_seconds=segment_seconds)
    return {
        "start_sec": start,
        "end_sec": end,
        "peak_bin": float(beta.argmax().item()),
        "peak_attention": float(beta.max().item()),
    }
