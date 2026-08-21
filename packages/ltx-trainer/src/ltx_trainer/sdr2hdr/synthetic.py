"""CRF/noise SDR synthesis and Tedla training pairs."""

from __future__ import annotations

import torch
from torch import Tensor

from ltx_trainer.hdr_ingest import synthetic_gamma_ldr_stack_from_linear_hdr
from ltx_trainer.sdr2hdr.mevm_proxy import DEFAULT_MEVM_EVS


def tedla_linear_brackets(hdr_cfhw: Tensor, ref_ev: float) -> tuple[Tensor, list[float]]:
    """Linear radiance brackets relative to ``ref_ev``."""
    evs = [float(e) for e in DEFAULT_MEVM_EVS]
    outs = [(hdr_cfhw * (2.0 ** (ev - ref_ev))).clamp(0.0, 1.0) for ev in evs]
    return torch.stack(outs, dim=0), evs


def _apply_crf_noise(sdr: Tensor, *, seed: int = 0) -> Tensor:
    g = torch.Generator(device=sdr.device)
    g.manual_seed(seed)
    noise = torch.randn(sdr.shape, generator=g, device=sdr.device, dtype=sdr.dtype) * 0.02
    quant = torch.round((sdr + noise).clamp(0, 1) * 255.0) / 255.0
    return quant.clamp(0.0, 1.0)


def synthesize_tedla_training_pair(
    hdr_cfhw: Tensor,
    *,
    gamma: float = 2.2,
    seed: int = 0,
) -> tuple[Tensor, Tensor, list[float]]:
    """Synthetic SDR clip + γ brackets from scene-linear HDR ``[C,F,H,W]``."""
    stack, evs = synthetic_gamma_ldr_stack_from_linear_hdr(
        hdr_cfhw,
        -4.0,
        4.0,
        4.0,
        gamma=gamma,
        normalize="p999",
    )
    mid = stack[len(evs) // 2]
    sdr = _apply_crf_noise(mid.pow(gamma), seed=seed).pow(1.0 / gamma)
    brackets, out_evs = tedla_linear_brackets(hdr_cfhw, ref_ev=0.0)
    brackets = brackets.pow(1.0 / gamma)
    return sdr, brackets, out_evs
