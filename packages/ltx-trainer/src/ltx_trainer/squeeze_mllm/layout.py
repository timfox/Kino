"""Architecture layout and limitations for Squeeze-MLLM (arXiv:2605.26111)."""

from __future__ import annotations

from typing import Any


def architecture_layout() -> dict[str, Any]:
    return {
        "encoders": {
            "mllm": "InternVL3-8B (frozen) — joint text + reference image tokens",
            "vae": "FLUX VAE encoder — fine-grained identity latents",
        },
        "aggregator": "Dual Layer Aggregator (DLA): separate LAP on text vs image layer stacks",
        "decoder": "FLUX.1-dev DiT with LoRA rank 512 on attention blocks",
        "inference": "Multi-stage denoising: MLLM early → MLLM+VAE middle → VAE late",
        "training": "Stage-1 MLLM→DiT only; Stage-2 MLLM+VAE joint",
    }


def paper_limitations() -> list[str]:
    return [
        "MLLM text space vs T5-conditioned DiT space is not fully aligned without large T2I pretraining.",
        "Multi-subject evaluation limited by scarce public multi-subject data (lightweight MUSAR fine-tune only).",
        "Reference stub does not run full FLUX/InternVL training or DreamBench inference.",
    ]
