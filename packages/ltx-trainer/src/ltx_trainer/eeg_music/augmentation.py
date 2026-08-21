"""Channel-wise data augmentation (Algorithm 2)."""

from __future__ import annotations

import torch
from torch import Tensor


def channel_drop(x: Tensor, drop_prob: float, *, generator: torch.Generator | None = None) -> Tensor:
    """Drop random electrodes; x is (B, C, L)."""
    if drop_prob <= 0.0:
        return x
    b, c, _ = x.shape
    keep = torch.rand(b, c, generator=generator) > drop_prob
    return x * keep.unsqueeze(-1).to(x.dtype)


def augment_eeg_segment(
    x: Tensor,
    *,
    crop_scale: float = 0.8,
    noise_std: float = 0.05,
    channel_drop_prob: float = 0.2,
) -> Tensor:
    """Algorithm 2 stub: channel drop → random crop → resize → Gaussian noise."""
    x = channel_drop(x, channel_drop_prob)
    b, c, length = x.shape
    crop_len = max(1, int(crop_scale * length))
    start = 0 if length <= crop_len else int(torch.randint(0, length - crop_len + 1, (1,)).item())
    cropped = x[:, :, start : start + crop_len]
    resized = torch.nn.functional.interpolate(cropped, size=length, mode="linear", align_corners=False)
    noise = torch.randn_like(resized) * noise_std
    return resized + noise
