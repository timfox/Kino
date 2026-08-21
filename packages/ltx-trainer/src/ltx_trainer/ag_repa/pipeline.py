"""AG-REPA pipeline cards and demos."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.dataset import dataset_card
from ltx_trainer.ag_repa.mock import evaluation_smoke
from ltx_trainer.ag_repa.profiles import synthetic_fog_profile, synthetic_lasp_profile
from ltx_trainer.ag_repa.tables import (
    table1_scd_dissociation,
    table10_k_sensitivity,
    table11_projection_heads,
    table12_supervision_source,
    table2_alignment_strategies,
    table3_selection_targets,
    table4_generalization,
    table5_cross_architecture,
    table6_efficiency,
    table7_fog_stability,
    table8_freeze_vs_refresh,
    table9_static_vs_adaptive,
)
from ltx_trainer.ag_repa.taxonomy import SelectionStrategy


def framework_card(cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    return {
        "name": "AG-REPA",
        "paper": f"arXiv:{c.paper_arxiv}",
        "full_name": "Attribution-Guided REPresentation Alignment",
        "backbone": "DiT Flow Matching (unified TTS + TTA)",
        "teachers": {"semantic": c.semantic_teacher, "acoustic": c.acoustic_teacher},
        "diagnostics": {
            "BiT-C": "dual-stream teacher cosine at token interface",
            "LASP": "shared-projection layer storage probe",
            "FoG-A": "forward-only gate ablation on velocity field",
        },
        "training": {
            "warmup_probe_steps": c.warmup_probe_steps,
            "top_k_layers": c.top_k_layers,
            "checkpoint_steps": c.train_checkpoint_steps,
            "loss": "LFM + λBiT·LBiT + Σ λk·(1−cos) on FoG-A top-K",
        },
        "key_insight": "Align causally dominant layers (FoG-A), not representation-rich deep layers (LASP)",
    }


def knowledge_card(cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    return {
        "paper_arxiv": c.paper_arxiv,
        "github": c.github_url,
        "scd": "Semantic storage (deep LASP) ≠ causal drivers (early FoG-A)",
        "datasets": dataset_card(c),
        "token_configs": [c.token_config_a, c.token_config_b],
        "results_vs_best_static_repa": {
            "speech_fad_reduction_pct": 18,
            "audio_fad_reduction_pct": 16,
        },
        "efficiency": {
            "fog_probe_wallclock_pct": "<0.5%",
            "convergence_speedup_vs_lasp_repa": "≈3.26×",
        },
        "limitations": [
            "Static layer set after warm-up (stable but may drift on very long runs)",
            "Pooled MLP heads matched to global Whisper/BEATs targets",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_scd": table1_scd_dissociation(),
        "table2_alignment": table2_alignment_strategies(),
        "table3_selection": table3_selection_targets(),
        "table4_generalization": table4_generalization(),
        "table5_cross_arch": table5_cross_architecture(),
        "table6_efficiency": table6_efficiency(),
        "table7_stability": table7_fog_stability(),
        "table8_freeze_refresh": table8_freeze_vs_refresh(),
        "table9_static_adaptive": table9_static_vs_adaptive(),
        "table10_k_sensitivity": table10_k_sensitivity(),
        "table11_projection": table11_projection_heads(),
        "table12_supervision": table12_supervision_source(),
    }


def evaluation_demo(*, strategy: str = "ag_repa", cfg: AgRepaConfig | None = None) -> dict[str, Any]:
    c = cfg or AgRepaConfig()
    smoke = evaluation_smoke(c)
    fog = synthetic_fog_profile(c, seed=c.random_seed)
    lasp = synthetic_lasp_profile(c, domain="semantic")
    strat_key = strategy
    if strat_key == "ag_repa":
        strat_key = SelectionStrategy.AG_REPA.value
    try:
        strat = SelectionStrategy(strat_key)
    except ValueError:
        strat = SelectionStrategy.AG_REPA
    return {
        "strategy": strat.value,
        "fog_profile_head": [{"layer": s.layer, "score": round(s.score, 4)} for s in fog[:5]],
        "lasp_profile_tail": [{"layer": s.layer, "score": round(s.score, 4)} for s in lasp[-5:]],
        "smoke_summary": {
            k: smoke[k]
            for k in (
                "scd_overlap_top3",
                "fog_top3_synthetic",
                "lasp_top3_synthetic",
                "paper_speech_fad",
                "paper_audio_fad",
                "loss",
            )
        },
    }
