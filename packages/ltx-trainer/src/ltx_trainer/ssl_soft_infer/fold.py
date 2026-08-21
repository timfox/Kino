"""AV-fold annotation hook."""

from __future__ import annotations

from typing import Any


def annotate_audio_save_data(meta: dict[str, Any]) -> dict[str, Any]:
    """Tag precomputed rows with soft-infer assignment metadata."""
    return {
        **meta,
        "ssl_soft_infer": {
            "train_assignment": "hard",
            "infer_assignment": "soft",
            "paper": "arXiv:2606.06806",
        },
    }
