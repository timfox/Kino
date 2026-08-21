"""NMM roadmap taxonomy smoke (arXiv:2605.25343)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nmm.config import NMMConfig
from ltx_trainer.nmm.nativity import fusion_formulas
from ltx_trainer.nmm.taxonomy import IOCategory, describe_io, io_formulas


def evaluation_smoke(cfg: NMMConfig | None = None) -> dict[str, Any]:
    c = cfg or NMMConfig()
    formulas = io_formulas()
    card = describe_io(IOCategory.M2M)
    fusion = fusion_formulas()
    return {
        "paper": c.paper_arxiv,
        "io_paradigms": len(formulas),
        "fusion_regimes": len(fusion),
        "m2m_focus": card.get("focus", ""),
    }
