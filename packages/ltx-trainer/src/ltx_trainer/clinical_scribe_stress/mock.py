"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clinical_scribe_stress.config import ClinicalScribeStressConfig
from ltx_trainer.clinical_scribe_stress.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_main_results,
)


def evaluation_smoke(cfg: ClinicalScribeStressConfig | None = None) -> dict[str, Any]:
    c = cfg or ClinicalScribeStressConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    ambient_15 = next(r for r in table2_main_results(c) if r["condition"] == "DEMAND Ambient 15 dB")
    ref = next(r for r in table2_main_results(c) if "Reference" in r["condition"])

    assert c.ambient_15db_wer_delta == 0.71
    assert round(ambient_15["wer"] - ref["wer"], 2) == c.ambient_15db_wer_delta
    assert ambient_15["unsafe"] > ref["unsafe"] * 1.9  # nearly doubled
    assert c.clean_wer < c.semantic_5db_wer
    assert c.mitig_semantic_5db_unsafe < c.semantic_5db_unsafe
    assert c.mitig_ambient_5db_unsafe < c.ambient_5db_unsafe
    assert demo["err_prop_demo"] >= 0.0
    assert len(benchmarks_bundle(c)["table2_main"]) == 7

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "clean_wer": c.clean_wer,
        "ambient_15db_wer_delta_pp": c.ambient_15db_wer_delta,
        "clean_unsafe_pct": c.clean_unsafe,
        "ambient_15db_unsafe_pct": c.ambient_15db_unsafe,
        "unsafe_near_double": c.ambient_15db_unsafe_ratio >= 1.9,
        "n_encounters": c.n_encounters,
    }
