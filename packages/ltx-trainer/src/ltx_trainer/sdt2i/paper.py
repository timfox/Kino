"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sdt2i.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PANORAMA_TRIGGER


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "components": [
            "MSTD: StitchDiffusion + MultiDiffusion mask fusion",
            "MPF: PanFusion dual-branch + MD (pano/pers/both)",
            "Bootstrap coupling + BG-only EPPA during bootstrap",
            f"LoRA trigger: {PANORAMA_TRIGGER!r}",
        ],
        "outputs": ["erp_panorama_latent", "layout_certainty"],
    }
