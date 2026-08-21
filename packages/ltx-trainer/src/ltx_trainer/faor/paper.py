"""Framework summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.faor.config import CODE_URL, PAPER_ARXIV, PAPER_TITLE, PAPER_URL, SAFE_BLOCKS


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "code": CODE_URL,
        "venue": "AAAI 2025",
        "components": [
            f"SAFE encoder ({SAFE_BLOCKS} blocks in paper; stub uses 4)",
            "ATFM: pixel-wise Md + semantic Ms affine modulation",
            "SGIF: geodesic resampling-then-representation + implicit MLP",
        ],
        "outputs": ["arbitrary_scale_hr_erp"],
    }
