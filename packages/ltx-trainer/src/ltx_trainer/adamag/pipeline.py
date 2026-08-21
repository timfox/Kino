"""Framework card and benchmarks for AdaMaG."""

from __future__ import annotations

from typing import Any

from ltx_trainer.adamag.config import AdamagConfig
from ltx_trainer.adamag.layout import CONSERVATION_EQ5, LIMITATIONS, PIPELINE_STEPS
from ltx_trainer.adamag.mock import evaluation_smoke
from ltx_trainer.adamag.tables import (
    table1_high_guidance,
    table1_optimal_guidance,
    table2_compbench_sd3,
    table3_hires_sd3,
    table4_beta_ablation,
    table5_gamma_ablation,
)


def framework_card(cfg: AdamagConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AdamagConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "method": "Adaptive Manifold Guidance (AdaMaG)",
        "authors": "Esmati, Hyung, Dadashzadeh, Choo, Mirmehdi (Bristol / KAIST)",
        "problem": (
            "CFG linearly extrapolates conditional/unconditional velocities, breaking "
            "probability conservation and causing saturation and off-manifold artifacts at high ω."
        ),
        "conservation_decomposition": CONSERVATION_EQ5,
        "solution": {
            "score_parallel_attenuation": f"β dampens g∥ (default β={cfg.beta_default})",
            "time_schedule": f"ω(t)=max(ω_min, ω_ref(1-t)^γ) (default γ={cfg.gamma_default})",
            "inference_cost": "Same NFE as CFG — plug-and-play",
        },
        "cfg_baseline": "v_cfg = v_u + ω(v_c - v_u)",
        "adamag_update": "v = v_u + ω(t)(g⊥ + β g∥),  n_t = a_t x - v_c",
        "models_evaluated": list(cfg.models),
        "baselines": list(cfg.baselines),
        "headline_sd3_optimal": {
            "fid_cfg": cfg.sd3_fid_cfg,
            "fid_ours": cfg.sd3_fid_ours,
            "sat_cfg": cfg.sd3_sat_cfg,
            "sat_ours": cfg.sd3_sat_ours,
        },
        "headline_sd3_high_omega": {
            "omega": cfg.sd3_high_guidance_omega,
            "fid_cfg": cfg.sd3_fid_cfg_high,
            "fid_ours": cfg.sd3_fid_ours_high,
        },
        "evaluation_protocol": {
            "prompts": "5000 COCO val @ 256²; CompBench; 1024² preference metrics",
            "solver": f"Euler, {cfg.solver_steps} steps, H100",
        },
        "pipeline_steps": list(PIPELINE_STEPS),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_optimal": table1_optimal_guidance(),
        "table1_high_guidance": table1_high_guidance(),
        "table2_compbench_sd3": table2_compbench_sd3(),
        "table3_hires_sd3": table3_hires_sd3(),
        "table4_beta": table4_beta_ablation(),
        "table5_gamma": table5_gamma_ablation(),
    }


def evaluation_demo() -> dict[str, Any]:
    return {"smoke": evaluation_smoke(), "defaults": {"beta": 0.1, "gamma": 4.0}}


def headline_results() -> dict[str, Any]:
    cfg = AdamagConfig()
    return {
        "sd3_fid_improvement_optimal": cfg.sd3_fid_cfg - cfg.sd3_fid_ours,
        "sd3_sat_reduction_optimal": cfg.sd3_sat_cfg - cfg.sd3_sat_ours,
        "no_extra_nfe": True,
    }
