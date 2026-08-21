"""CryAcc evaluation smoke (arXiv:2605.28687)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cryacc.config import CryAccConfig
from ltx_trainer.cryacc.icc import icc_rating
from ltx_trainer.cryacc.pipeline import pipeline_demo


def evaluation_smoke(cfg: CryAccConfig | None = None) -> dict[str, Any]:
    c = cfg or CryAccConfig()
    demo = pipeline_demo(c, seed=42)
    return {
        "paper": c.paper_arxiv,
        "n_infants": c.n_infants_total,
        "n_measures": 7,
        "f0_icc_anchor": c.icc_f0,
        "f0_icc_computed": demo.get("f0_icc_computed"),
        "f0_excellent": demo["icc"]["f0_excellent"],
        "icc_smoke": demo["icc"]["f0_excellent"],
        "computed_from_cohort": demo["icc"].get("computed_from_cohort", False),
    }
