"""LTX-native MEVM finetune plan (Tedla arXiv:2605.14703 on Wan2.2-I2V-5B analogue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LtxMevmPlan:
    """How to adapt GOPEX LTX-AV for exposure-bracket video generation."""

    base: str = "ltx-av fold + Gemma4 text; Wan MEVM weights not portable"
    temporal_concat: str = "latent [V_crf | V0 | V- | V+] along frame axis (paper Eq. 4)"
    exposure_rope: str = "ltx_trainer.sdr2hdr.exposure_rope.ExposureRoPEOffset on DiT RoPE"
    train_data: str = "HDR preprocess shards (hdr_latent) + synthetic CRF SDR input"
    loss: str = "flow-matching L1 on bracket latents only; CRF conditioning frames clean"
    vmm: str = "train separately via scripts/train_sdr2hdr_vmm.py; inference merge=vmm"
    delivery_default: str = "photometric MEVM proxy until MEVM LoRA exists"
    arxiv: str = "2605.14703"


def mevm_ltx_training_notes() -> dict[str, Any]:
    return {
        "paper": "Generating HDR Video from SDR Video (Tedla et al.)",
        "arxiv": "2605.14703",
        "stages": [
            "1. HDR ingest → scene-linear hdr_latent in precomputed .pt",
            "2. VMM train: ./scripts/kino-sdr2hdr.sh train-vmm [precomputed-hdr/] -o vmm.pt",
            "3. (Future) MEVM LoRA: temporal concat + ExposureRoPEOffset in trainer",
            "4. Delivery: GOPEX_SDR2HDR_LIFT_MODE=merge GOPEX_SDR2HDR_VMM=vmm.pt",
        ],
        "proxy_until_mevm": "mevm_photometric_brackets (no generative highlight inpainting)",
        "env": {
            "GOPEX_SDR2HDR_LIFT_MODE": "merge | tonemap_only | vdp",
            "GOPEX_SDR2HDR_MERGE": "vmm | debevec",
            "GOPEX_SDR2HDR_VMM": "path to train_sdr2hdr_vmm.pt",
            "GOPEX_SDR2HDR_VAE_ROUNDTRIP": "1 applies simulate_vae_roundtrip at inference",
        },
    }
