"""Toy smoke hooks for validate_paper_stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.swansphere.pipeline import pipeline_demo


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = pipeline_demo()
    return {
        "num_patches": demo["num_patches"],
        "svac_loss_full": round(demo["svac_loss_full"], 4),
        "swansphere_fd": demo["swansphere_fd"],
        "first_chunk_s": demo["framework"]["latency"]["first_chunk_total_s"],
    }
