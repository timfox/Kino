"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spherefusion.config import FPS_512, INFERENCE_SEC_512, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "components": [
            "ERP ResNet50 image encoder + icosahedral mesh ResNet18 encoder",
            "GateFuse: GRU-style gates fuse Feq→sphere with Fsp",
            "Mesh decoder with FAF cache for fast inference",
            f"~{FPS_512:.0f} FPS @ 512×1024 ({INFERENCE_SEC_512*1000:.1f} ms on RTX 3090)",
        ],
        "outputs": ["panorama_depth_erp", "depth_mesh"],
    }
