"""Framework card and benchmark tables."""

from __future__ import annotations

from typing import Any

from ltx_trainer.eisbach.analysis import analysis_demo
from ltx_trainer.eisbach.barrier import barrier_demo
from ltx_trainer.eisbach.config import EisbachConfig
from ltx_trainer.eisbach.dora import dora_demo


def framework_card(cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    return {
        "paper": cfg.paper_id,
        "title": cfg.title,
        "framework": cfg.framework,
        "backbone": cfg.backbone,
        "components": [
            "eisbach_log_barrier",
            "dit_belief_entropy",
            "implicit_online_curriculum",
            "dora_structural_disentanglement",
        ],
        "barrier_lambda": cfg.barrier_lambda,
        "adapter": cfg.adapter,
        "headline": {
            "barrier_dynamic_range_db": cfg.barrier_dynamic_range_db,
            "baseline_dynamic_range_db": cfg.baseline_dynamic_range_db,
        },
    }


def table1_curation_comparison() -> list[dict[str, Any]]:
    """Table 1 — explicit vs implicit curation."""
    return [
        {"property": "When", "explicit": "Before training, offline", "eisbach": "During training, online"},
        {"property": "Criterion", "explicit": "Human or external model", "eisbach": "DiT forward pass entropy"},
        {"property": "Adaptive?", "explicit": "No (fixed)", "eisbach": "Yes (model converges)"},
        {"property": "External signal?", "explicit": "Yes", "eisbach": "No"},
        {"property": "Discarded samples", "explicit": "Never trained", "eisbach": "Damped gradients only"},
    ]


def table2_structural_dimensions() -> list[dict[str, Any]]:
    """Table 2 — barrier vs baseline structural dimensions."""
    return [
        {"dimension": "Self-similarity block scale", "barrier": "Mid-scale blocks, regular dark zones", "baseline": "Large-scale diffuse brightness"},
        {"dimension": "Spectral peak-to-valley contrast", "barrier": "Sharp peaks, deep valleys", "baseline": "Flat spectrum"},
        {"dimension": "Spectral flux distribution", "barrier": "Sparse peaks at transitions", "baseline": "Continuous low fluctuation"},
        {"dimension": "PCA trajectory coverage", "barrier": "Large area, distinct clusters", "baseline": "Small area, gradual drift"},
        {"dimension": "Dynamic range", "barrier": ">40 dB", "baseline": "<25 dB"},
    ]


def table_training_config() -> dict[str, Any]:
    """§7 training hyperparameters."""
    cfg = EisbachConfig()
    return {
        "backbone": cfg.backbone,
        "dataset": cfg.dataset,
        "adapter": cfg.adapter,
        "rank": cfg.lora_rank,
        "alpha": cfg.lora_alpha,
        "steps": cfg.train_steps,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "barrier_lambda": cfg.barrier_lambda,
        "baseline_lambda": cfg.baseline_lambda,
    }


def testable_predictions() -> list[dict[str, Any]]:
    """§9 — five testable predictions."""
    cfg = EisbachConfig()
    return [
        {"id": 1, "prediction": "Low-t w→1 high gradient; high-t persistent damping"},
        {"id": 2, "prediction": f"Optimal λ ≈ {cfg.optimal_lambda_low}–{cfg.optimal_lambda_high}; collapse beyond {cfg.lambda_collapse}"},
        {"id": 3, "prediction": "Higher cross-seed structure consistency + detail diversity"},
        {"id": 4, "prediction": "LoRA-only reduces PCA coverage vs DoRA+barrier"},
        {"id": 5, "prediction": "Low-t B-loss correlated; high-t correlation drops/inverts"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_curation": table1_curation_comparison(),
        "table2_structural_dimensions": table2_structural_dimensions(),
        "training_config": table_training_config(),
        "testable_predictions": testable_predictions(),
    }


def headline_results(cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    return {
        "barrier_lambda": cfg.barrier_lambda,
        "dynamic_range_gain_db": cfg.barrier_dynamic_range_db - cfg.baseline_dynamic_range_db,
        "n_characters": len(cfg.characters),
        "train_steps": cfg.train_steps,
    }


def evaluation_demo(*, seed: int = 0, cfg: EisbachConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EisbachConfig()
    return {
        "framework": framework_card(cfg),
        "barrier": barrier_demo(seed=seed, cfg=cfg),
        "dora": dora_demo(seed=seed, cfg=cfg),
        "analysis": analysis_demo(seed=seed, cfg=cfg),
        "headline": headline_results(cfg),
    }
