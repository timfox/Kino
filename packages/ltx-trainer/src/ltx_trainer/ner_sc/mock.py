"""NeR-SC evaluation smoke for paper stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ner_sc.pipeline import pipeline_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = pipeline_demo()
    return {
        "psnr_db": demo["psnr_db"],
        "ms_ssim": demo["ms_ssim"],
        "skip_rate": demo["skip"]["skip_rate"],
        "mgf_channels": demo["mgf_channels"],
    }
