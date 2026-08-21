"""Framework metadata."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3po.config import (
    NUM_DUAL_DUCT_BLOCKS,
    NUM_INPUT_FRAMES,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    PARAMS_M,
    SCALE_FACTOR,
)
from ltx_trainer.s3po.pipeline import evaluation_demo_run


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "components": {
            "360_feature_extractor": "Co-joint Ft−1,Ft,Ft+1 + spatial/channel attention (Eq. 1–2)",
            "recurrent_global": "Ft ⊕ ht−1 ⊕ (HRt−1)↓ pixel-unshuffle (Eq. 3)",
            "dual_duct": f"{NUM_DUAL_DUCT_BLOCKS}× mutual-exchange residual blocks (Eq. 4–6)",
            "upsample": "Dual pixel-shuffle residues + bicubic baseline (Fig. 8)",
            "loss": "Weighted Spherically Smooth-L1 WSS-L1 (Eq. 7)",
            "dataset": "360VDS (590 clips, 480×360 ERP)",
        },
        "scale": SCALE_FACTOR,
        "params_m": PARAMS_M,
        "input_frames": NUM_INPUT_FRAMES,
        "alignment_free": True,
    }


def evaluation_demo() -> dict[str, Any]:
    return {"package": "s3po", **evaluation_demo_run()}
