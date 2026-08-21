"""Framework card and survey bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.energy_aware.config import EnergyAwareConfig
from ltx_trainer.energy_aware.hardware import AI_ACCELERATORS_2026, exascale_table_dict
from ltx_trainer.energy_aware.ltx_training import ltx_training_energy_plan
from ltx_trainer.energy_aware.mock import evaluation_smoke
from ltx_trainer.energy_aware.taxonomy import CORPUS_KEYWORD_TOP, taxonomy_dict


def framework_card(cfg: EnergyAwareConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EnergyAwareConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Exascale AI and GenAI push cloud-edge-HPC power/carbon past the Dennard "
            "power wall; sustainability needs cross-stack hardware, software, and policy."
        ),
        "method": {
            "taxonomy": "Seven pillars: profiling, hardware, software, DPM, scheduling, green AI, cooling",
            "metrics": "PUE, WUE, kg CO2e/kWh, joules/token, TDP-aware facility caps",
            "scheduling": "Carbon-aware temporal/spatial shifting, VM consolidation, power capping",
            "green_ai": "Training elasticity, inference prefill/decode DVFS, compression trade-offs",
        },
        "keywords": list(cfg.keywords),
        "liquid_cooling_threshold_kw": cfg.liquid_cooling_rack_kw_threshold,
    }


def exascale_operating_costs() -> dict[str, Any]:
    return {"exascale_top500_nov2025": exascale_table_dict()}


def taxonomy_overview() -> dict[str, Any]:
    return taxonomy_dict()


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "corpus_keywords_top": list(CORPUS_KEYWORD_TOP),
        "ai_accelerators_2026": [
            {
                "name": a.name,
                "tdp_w": a.tdp_w,
                "precision": a.peak_precision,
                "memory": a.memory,
            }
            for a in AI_ACCELERATORS_2026
        ],
        "literature_share_software": {"consolidation_scheduling_pct": 46, "dvfs_capping_pct": 31, "code_compiler_pct": 23},
        "literature_share_green_ai": {"inference_rag_pct": 42, "edge_fl_pct": 31, "training_pct": 27},
        "literature_share_profiling": {"estimators_pct": 45, "silicon_ebpf_pct": 34, "out_of_band_pct": 21},
    }


def evaluation_demo() -> dict[str, Any]:
    return {
        "smoke": evaluation_smoke(),
        "exascale": exascale_operating_costs(),
        "taxonomy": taxonomy_overview(),
        "ltx_training_plan_8gpu_100h": ltx_training_energy_plan(num_gpus=8, hours=100.0),
    }
