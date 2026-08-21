"""Framework summary for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ulayout.config import GITHUB_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "code": GITHUB_URL,
        "components": [
            "ERP unification + vertical shift by pitch (Sec. 3.1–3.2)",
            "Dual-branch ResNet-50 + 1D conv efficient extractor (Sec. 3.3.1)",
            "SWG-Transformer boundary regression (LGT-Net, Sec. 3.3.2)",
            "Ceiling/floor 1D boundaries; pano L_b+L_d+L_n+L_g, perspective L_b (Sec. 3.3.3)",
        ],
        "outputs": ["ceiling_boundary", "floor_boundary", "horizon_depth_pano"],
    }
