"""Reference metrics from LumaFlux paper (arXiv:2604.02787)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lumaflux.config import LORA_RANK, TRAIN_ITERATIONS
from ltx_trainer.lumaflux.metrics import (
    TABLE1_BENCHMARKS,
    TABLE2_TMO,
    TABLE3_ABLATION,
    TABLE4_USER_STUDY,
)

PAPER_ARXIV = "2604.02787"
PAPER_TITLE = "LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers"

TRAINING_DEFAULTS = {
    "backbone": "Flux MM-DiT (frozen)",
    "adapters": "PGA + PCM + HDR Residual Coupler + RQS",
    "train_iterations": TRAIN_ITERATIONS,
    "batch_size": 16,
    "lr": 1e-4,
    "lora_rank": LORA_RANK,
    "inference_steps": 40,
    "encoding": "PQ BT.2020 10-bit",
    "peak_nits": 1000,
    "prompt_free": True,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_benchmarks": TABLE1_BENCHMARKS,
        "table2_tmo_luma_eval": TABLE2_TMO,
        "table3_ablation": TABLE3_ABLATION,
        "table4_user_study": TABLE4_USER_STUDY,
        "training_defaults": TRAINING_DEFAULTS,
    }
