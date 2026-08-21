"""End-to-end SDR video → HDR (MEVM proxy + VMM / Debevec merge)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ltx_trainer.hdr_ingest import soft_knee_compress_linear
from ltx_trainer.sdr2hdr.mevm_proxy import DEFAULT_MEVM_EVS, mevm_photometric_brackets, simulate_vae_roundtrip
from ltx_trainer.sdr2hdr.vmm import VideoMergingModel, load_vmm_checkpoint, merge_brackets_debevec


@dataclass
class Sdr2HdrConfig:
    merge: str = "debevec"
    device: str = "cpu"
    evs: tuple[float, ...] = DEFAULT_MEVM_EVS
    gamma: float = 2.2
    vmm_path: str | None = None
    temporal_smooth: bool = True
    vae_roundtrip: bool = False
    input_shoulder: bool = False


def sdr_gamma_soft_shoulder(
    sdr_gamma_cfhw: Tensor,
    *,
    threshold: float = 0.92,
    softness: float = 0.08,
) -> Tensor:
    """Soft shoulder on γ-encoded SDR before bracketing."""
    linear = sdr_gamma_cfhw.clamp(0.0, 1.0).pow(2.2)
    compressed = soft_knee_compress_linear(linear, threshold=threshold, softness=softness)
    return compressed.pow(1.0 / 2.2)


def _robust_quantile(x: Tensor, q: float, *, max_samples: int = 1_048_576) -> Tensor:
    flat = x.reshape(-1)
    if flat.numel() > max_samples:
        step = max(1, flat.numel() // max_samples)
        flat = flat[::step][:max_samples]
    return torch.quantile(flat.float(), float(q))


def tone_map_hdr_for_preview(
    hdr_cfhw: Tensor,
    *,
    mode: str = "filmic",
    percentile: float = 0.99,
) -> Tensor:
    """Map scene-linear HDR to ``[0,1]`` preview (subsampling for large tensors)."""
    x = hdr_cfhw.detach().float().clamp(min=0.0)
    peak = _robust_quantile(x, percentile).clamp(min=1e-6)
    if mode == "reinhard":
        mapped = x / (x + peak)
    else:
        mapped = x / (x + peak)
    return mapped.clamp(0.0, 1.0)


def sdr_video_to_hdr(
    sdr_gamma_cfhw: Tensor,
    config: Sdr2HdrConfig | None = None,
) -> tuple[Tensor, Tensor, list[float]]:
    cfg = config or Sdr2HdrConfig()
    sdr = sdr_gamma_cfhw
    if cfg.input_shoulder:
        sdr = sdr_gamma_soft_shoulder(sdr)
    brackets, evs = mevm_photometric_brackets(
        sdr,
        evs=cfg.evs,
        gamma=cfg.gamma,
        temporal_smooth=cfg.temporal_smooth,
    )
    if cfg.vae_roundtrip:
        brackets = simulate_vae_roundtrip(brackets)

    merge = cfg.merge.lower().strip()
    if merge == "vmm":
        model = (
            load_vmm_checkpoint(cfg.vmm_path, device=cfg.device)
            if cfg.vmm_path
            else VideoMergingModel(num_exposures=len(evs)).to(cfg.device)
        )
        model.eval()
        with torch.inference_mode():
            hdr = model(brackets.to(cfg.device), evs).cpu()
    elif merge == "debevec":
        hdr = merge_brackets_debevec(brackets, evs, gamma=cfg.gamma)
    else:
        raise ValueError(f"Unknown merge mode {cfg.merge!r} (use debevec or vmm)")

    return hdr, brackets, evs
