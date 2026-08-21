"""Framework summary for agents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.edm.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, PROJECT_URL


def framework_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "project": PROJECT_URL,
        "components": [
            "Multi-scale encoder (DKM-style ResNet50 stub)",
            "SSAM: GP kernel + 3D spherical positional embedding",
            "Geodesic flow refinement via π / π⁻¹ ERP projection",
            "Angular cosine + certainty loss on unit sphere",
            "Azimuth rotation augmentation for ERP",
        ],
        "outputs": ["dense_match_cartesian", "certainty_map"],
    }
