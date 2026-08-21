"""Fold WaveNeXt 2 spectral envelope probe into audio latents."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.wavenext2.config import Wavenext2Config
from ltx_trainer.wavenext2.stft import stft_spec_features


def wavenext2_meta_block() -> dict[str, Any]:
    return {
        "wavenext2": {
            "arxiv_id": "2605.25506",
            "fold_role": "audio_spectral_envelope_probe",
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    cfg = Wavenext2Config()
    latents = data.get("latents")
    out = dict(data)
    out.update(wavenext2_meta_block())
    if latents is None:
        return out

    arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 8:
        flat = np.pad(flat, (0, 8 - flat.size))

    spec = stft_spec_features(flat[: min(256, flat.size)], n_fft=64)
    band_energy = float(np.abs(spec).mean())

    out["wavenext2"].update(
        {
            "stft_band_energy": round(band_energy, 6),
            "bddm_steps": len(cfg.bddm_noise_schedule),
            "convnext_blocks": cfg.convnext_blocks,
        }
    )
    return out
