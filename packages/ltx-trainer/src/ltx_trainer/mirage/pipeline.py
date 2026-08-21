"""Mirage framework card and evaluation demos."""
from __future__ import annotations

from typing import Any

from ltx_trainer.mirage.ablation import run_ablation_smoke
from ltx_trainer.mirage.benchmarks import benchmarks_bundle
from ltx_trainer.mirage.config import MirageConfig
from ltx_trainer.mirage.efficiency import efficiency_demo
from ltx_trainer.mirage.ltx_plan import ltx_training_plan
from ltx_trainer.mirage.rollout import run_toy_rollout
from ltx_trainer.mirage.training import run_toy_training


def framework_card(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    return {
        "name": cfg.name,
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "website": cfg.website,
        "backbone": cfg.backbone,
        "memory": "latent_spatial_cache_M_eq_pi_fi",
        "ltx_hook": "ControlNet-style latent readout side branch on Wan2.2 TI2V",
    }


def knowledge_card(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    from ltx_trainer.mirage.algorithm import algorithm1_steps

    return {
        "framework": framework_card(cfg),
        "benchmarks": benchmarks_bundle(),
        "ltx_plan": ltx_training_plan(cfg),
        "algorithm1": algorithm1_steps(cfg),
    }


def paper_checks() -> dict[str, bool]:
    ws = benchmarks_bundle()["worldscore"]
    re10k = benchmarks_bundle()["realestate10k"]
    return {
        "worldscore_sota_avg": ws["Mirage"]["average"] >= ws["Spatia"]["average"],
        "worldscore_3d_cons_leads": ws["Mirage"]["3d_cons"] >= ws["Spatia"]["3d_cons"],
        "re10k_ssim_best": re10k["Mirage"]["ssim"] >= max(m["ssim"] for m in re10k.values()),
        "closed_loop_psnr_c_best": re10k["Mirage"]["psnr_c"] >= max(m.get("psnr_c", 0) for m in re10k.values()),
    }


def evaluation_demo(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    return {
        "rollout": run_toy_rollout(cfg, n_chunks=3, seed=3),
        "training": run_toy_training(cfg, seed=3),
        "efficiency": efficiency_demo(cfg),
        "ablation": run_ablation_smoke(cfg),
        "paper_tables": benchmarks_bundle(),
        "paper_checks": paper_checks(),
    }


def evaluation_smoke(cfg: MirageConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MirageConfig()
    ev = evaluation_demo(cfg)
    checks = ev["paper_checks"]
    return {
        "package": "ltx_trainer.mirage",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "ok": all(checks.values()),
        "checks": checks,
        "memory_size": ev["rollout"]["memory_size"],
    }
