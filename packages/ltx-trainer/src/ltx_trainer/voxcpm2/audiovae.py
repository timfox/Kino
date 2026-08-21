"""AudioVAE V2 asymmetric codec stub (§3.2, Table 10)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voxcpm2.config import Voxcpm2Config


def audiovae_v2_params(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    return {
        "encode_hz": c.encode_sample_rate_hz,
        "decode_hz": c.decode_sample_rate_hz,
        "latent_dim": c.latent_dim,
        "latent_fps": c.latent_fps,
        "encoder_downsample": [2, 5, 8, 8],
        "decoder_upsample": [8, 6, 5, 2, 2, 2],
        "patch_frames": c.patch_frames,
        "lm_token_rate_hz": c.lm_token_rate_hz,
        "implicit_super_resolution": True,
    }


def table10_reconstruction() -> list[dict[str, Any]]:
    """Table 10 — VCTK / Song Describer reconstruction anchors."""
    return [
        {
            "model": "VoxCPM",
            "input_hz": 16000,
            "output_hz": 16000,
            "meld_48k_vctk": 1.787,
            "meld_16k_vctk": 0.801,
            "stoi_16k_vctk": 0.911,
            "pesq_16k_vctk": 3.940,
        },
        {
            "model": "VoxCPM1.5",
            "input_hz": 44100,
            "output_hz": 44100,
            "meld_48k_vctk": 1.139,
            "meld_16k_vctk": 0.926,
            "stoi_16k_vctk": 0.836,
            "pesq_16k_vctk": 3.148,
        },
        {
            "model": "VoxCPM2",
            "input_hz": 16000,
            "output_hz": 48000,
            "meld_48k_vctk": 1.335,
            "meld_16k_vctk": 0.813,
            "stoi_16k_vctk": 0.907,
            "pesq_16k_vctk": 3.906,
        },
    ]


def audiovae_demo(cfg: Voxcpm2Config | None = None) -> dict[str, Any]:
    c = cfg or Voxcpm2Config()
    v2 = next(r for r in table10_reconstruction() if r["model"] == "VoxCPM2")
    return {
        "params": audiovae_v2_params(c),
        "vctk_meld_48k": v2["meld_48k_vctk"],
        "matches_config": v2["meld_48k_vctk"] == c.vae_meld_48k_vctk,
    }
