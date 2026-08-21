"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cubediff.config import (
    FACE_FOV_TRAIN_DEG,
    NUM_FACES,
    OVERLAP_DEG,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PROJECT_URL,
    TRAIN_ITERS,
)
from ltx_trainer.cubediff.integration import proceduralsky_card


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "venue": "ICLR 2025",
        "components": [
            f"{NUM_FACES}-face cubemap parallel diffusion",
            "Inflated self/cross-attention across faces (T×HW tokens)",
            "Synchronized GroupNorm in VAE",
            "Cube UV positional encoding + 2.5° overlap crop",
            f"Training: {TRAIN_ITERS} iters, v-prediction, DDIM-50",
        ],
        "face_fov": f"{FACE_FOV_TRAIN_DEG}° train / 90° crop (±{OVERLAP_DEG}° overlap)",
        "integration": (
            "GOPEX stub: cubemap I/O, inflated attention, Table 1–2 anchors. "
            "Generated ERP panoramas can feed HDR sky / IBL workflows "
            f"(e.g. {proceduralsky_card()['partner']}) alongside catalog environment maps."
        ),
        "proceduralsky": proceduralsky_card(),
    }
