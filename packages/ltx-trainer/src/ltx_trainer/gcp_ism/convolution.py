"""FFT geometric convolution for GCP / GCP-ISM (Eqs. 7–8, 14–15)."""

from __future__ import annotations

import torch
from torch import Tensor


def geometric_convolve_1d(f: Tensor, prev: Tensor) -> Tensor:
    """Multiply in frequency domain: F[h_N] = F[f_N] ⊙ F[h_{N-1}] (Eq. 8)."""
    n = max(f.numel(), prev.numel())
    size = 1
    while size < n:
        size *= 2
    ff = torch.fft.rfft(f, n=size)
    fp = torch.fft.rfft(prev, n=size)
    out = torch.fft.irfft(ff * fp, n=size)
    return out[: prev.numel()].real


def build_square_kernel(k2_max: int) -> Tensor:
    """Kernel with 1 at τ=0 and 2 at perfect squares up to k2_max."""
    f = torch.zeros(k2_max + 1)
    f[0] = 1.0
    n = 1
    while n * n <= k2_max:
        f[n * n] = 2.0
        n += 1
    return f
