"""Fast Fourier Convolution stub (§II-B)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fsc_net.config import FscNetConfig


def fast_fourier_conv2d(x: np.ndarray) -> np.ndarray:
    """Y = Conv_local(X) + IFFT(Conv_global(FFT(X))) stub."""
    x = np.asarray(x, dtype=np.float64)
    local = x * 0.5
    spec = np.fft.rfft2(x)
    global_branch = np.fft.irfft2(spec * 0.5, s=x.shape[-2:])
    return local + global_branch


def channel_wise_subband(x: np.ndarray, *, num_subbands: int = 3) -> np.ndarray:
    """Partition frequency axis into B subbands stacked on channel dim."""
    x = np.asarray(x, dtype=np.float64)
    t, f = x.shape[-2], x.shape[-1]
    band = max(1, f // num_subbands)
    bands = [x[..., i * band : (i + 1) * band] for i in range(num_subbands)]
    return np.stack(bands, axis=0)


def ffc_demo(*, seed: int = 0, cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    rng = np.random.default_rng(seed)
    feat = rng.standard_normal((8, 32, 64))
    y = fast_fourier_conv2d(feat)
    cws = channel_wise_subband(feat[0], num_subbands=cfg.num_subbands)
    return {
        "input_shape": list(feat.shape),
        "output_shape": list(y.shape),
        "cws_shape": list(cws.shape),
        "num_subbands": cfg.num_subbands,
        "global_receptive_field": "full_spectrum",
    }
