"""Framework metadata."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dense360.config import (
    BASELINE_VLM,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    SEG_TOKEN,
)
from ltx_trainer.dense360.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "components": {
            "dataset": "160K ERP panoramas; 5M entity captions; 1M ref-seg; 100K GCG descriptions",
            "bench": "Dense360-Bench: 3K grounding + 3K captioning on 1.3K ERP images",
            "erp_rope": "Horizontal f(w) + latitude γ scaling for mRoPE extension (Eq. 1–4)",
            "dense360_vlm": f"{BASELINE_VLM} + {SEG_TOKEN} mask decoding",
            "level1": "4 slice views + CropFormer masks merged on ERP",
            "level2": "Tag + CoT captions + SAM verification IoU reliability",
            "level3": "Entity-grounded panoramic scene descriptions (GPT-4o)",
        },
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "dense360", **evaluation_demo_run()}
