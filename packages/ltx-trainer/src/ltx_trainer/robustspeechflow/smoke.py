"""Torch-free evaluation_smoke for RobustSpeechFlow."""

from __future__ import annotations


def evaluation_smoke() -> dict[str, object]:
    return {
        "lambda_rand": 0.2,
        "lambda_aug": 0.2,
        "torch_available": False,
        "paper_note": "full augment+contrastive smoke requires torch",
    }
