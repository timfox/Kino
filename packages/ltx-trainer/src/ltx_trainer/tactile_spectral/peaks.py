"""Spectral peak selection (sPeak, Sec. 4.4.3)."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.tactile_spectral.spectrum import band_mask


def select_spectral_peaks(
    mag: Tensor,
    freqs: Tensor,
    *,
    num_peaks: int = 10,
    jnd_fraction: float = 0.12,
    f_low: float = 20.0,
    f_high: float = 1000.0,
) -> tuple[Tensor, Tensor]:
    """Pick peaks separated by ~12% JND in frequency (Fielder & Vardar)."""
    mask = band_mask(freqs, f_low, f_high)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(0, device=mag.device), torch.zeros(0, device=mag.device)
    m = mag[idx]
    f = freqs[idx]
    order = torch.argsort(m, descending=True)
    chosen_f: list[float] = []
    chosen_a: list[float] = []
    for i in order.tolist():
        fi = float(f[i].item())
        ai = float(m[i].item())
        if not chosen_f:
            chosen_f.append(fi)
            chosen_a.append(ai)
        else:
            ok = all(abs(fi - cf) / max(cf, 1e-6) >= jnd_fraction for cf in chosen_f)
            if ok:
                chosen_f.append(fi)
                chosen_a.append(ai)
        if len(chosen_f) >= num_peaks:
            break
    return (
        torch.tensor(chosen_f, device=mag.device, dtype=freqs.dtype),
        torch.tensor(chosen_a, device=mag.device, dtype=mag.dtype),
    )
