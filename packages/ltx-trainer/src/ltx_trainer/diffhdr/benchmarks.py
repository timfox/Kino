"""Reference metrics from DiffHDR paper (arXiv:2604.06161)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.diffhdr.config import (
    HDRI_COUNT,
    LORA_RANK,
    NUM_FRAMES,
    TOTAL_SEQUENCES,
    TRAIN_STEPS,
)
from ltx_trainer.diffhdr.metrics import (
    TABLE1_SI_HDR,
    TABLE2_VIDEO,
    TABLE3_IN_THE_WILD,
    TABLE4_LOG_GAMMA,
    TABLE5_ABLATION,
)

PAPER_ARXIV = "2604.06161"
PAPER_TITLE = "DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models"

TRAINING_DEFAULTS = {
    "backbone": "Wan-2.1-VACE-14B",
    "vae": "Wan-2.1-VAE (4×8×8, frozen FP32 decode)",
    "resolution": "33×1280×720",
    "lora_rank": LORA_RANK,
    "train_steps": TRAIN_STEPS,
    "lr": 1e-4,
    "hdri_sources": HDRI_COUNT,
    "sequences": TOTAL_SEQUENCES,
    "frames_per_sequence": NUM_FRAMES,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_si_hdr": TABLE1_SI_HDR,
        "table2_video": TABLE2_VIDEO,
        "table3_in_the_wild": TABLE3_IN_THE_WILD,
        "table4_log_gamma_vae": TABLE4_LOG_GAMMA,
        "table5_ablation": TABLE5_ABLATION,
        "training_defaults": TRAINING_DEFAULTS,
    }
