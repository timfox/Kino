"""UNIVID framework card, tables, and demos (arXiv:2606.05748)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.univid.capbench import capbench_catalog
from ltx_trainer.univid.config import UnividConfig
from ltx_trainer.univid.layout import LIMITATIONS
from ltx_trainer.univid.mock import evaluation_smoke, run_lite_rag_smoke, run_risk_filter_smoke
from ltx_trainer.univid.prompts import prompt_bundle


def framework_card(cfg: UnividConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UnividConfig()
    return {
        "name": "UNIVID — Unified Vision-Language Model for Video Moderation",
        "paper": cfg.paper_arxiv,
        "authors": "Yang, Zhang, Du, Zhang, Zheng, Zhao, Xiao, Liang, Xiao (ByteDance)",
        "task": "Policy-aware video captioning + three-stage industrial moderation pipeline",
        "idea": (
            "LLaVA-OV captioner replaces 1000+ black-box policy classifiers with interpretable "
            "captions; Risk Filter → UNIVID-Lite / UNIVID-RAG → Trend Governance on cached embeddings."
        ),
        "architecture": {
            "backbone_7b": cfg.backbone_7b,
            "backbone_1b": cfg.backbone_1b,
            "vision_encoder": cfg.vision_encoder,
            "max_frames": cfg.max_frames,
            "caption_max_words": cfg.caption_max_words,
        },
        "training": {
            "stages": list(cfg.training_stages),
            "samples_m": cfg.training_samples_m,
            "hardware": cfg.training_gpus,
        },
        "pipeline_stages": [
            "A Risk Filter — fusion embedding + policy heads",
            "B Moderation Actor — UNIVID-Lite + UNIVID-RAG (VKB top-3)",
            "C Trend Governance — few-shot trend head on cached embeddings",
        ],
        "capbench": capbench_catalog(),
        "production": {
            "leakage_reduction_pct": cfg.leakage_reduction_pct,
            "overkill_reduction_pct": cfg.overkill_reduction_pct,
            "brand_ads_match_pct": cfg.brand_ads_match_pct,
            "cost_usd_per_1m": cfg.deployment_cost_usd_per_1m,
            "qps_per_h100": cfg.deployment_qps_per_h100,
        },
        "defaults": cfg.__dict__,
    }


def table_i_training_stages() -> list[dict[str, str | float]]:
    """Table 1 — UNIVID training stages."""
    return [
        {"stage": "PT", "data_source": "LLaVA pretrain + internal", "type": "Single sentence caption", "samples_m": 1.6},
        {"stage": "FT", "data_source": "Synthetic + human-refined", "type": "Caption", "samples_m": 3.2},
        {"stage": "FT", "data_source": "Synthetic VQA", "type": "General VQA", "samples_m": 2.0},
        {"stage": "CFT", "data_source": "Hybrid caption", "type": "Caption", "samples_m": 0.1},
    ]


def table_ii_capbench_comparison() -> list[dict[str, str | bool | int]]:
    """Table 2 — CapBench vs prior caption benchmarks."""
    return [
        {"benchmark": "Dream-1k", "total": 1000, "violative": 0, "human_verified": True, "global": False, "uni_scoring": False},
        {"benchmark": "KuaiMod", "total": 1000, "violative": 422, "human_verified": True, "global": False, "uni_scoring": False},
        {"benchmark": "CapBench (Ours)", "total": 17210, "violative": 11476, "human_verified": True, "global": True, "uni_scoring": True},
    ]


def table_iii_capbench_results() -> list[dict[str, float | str]]:
    """Table 3 — CapBench evaluation (violative domain recall %)."""
    rows = [
        ("GPT-4.1", 45.9, 17.4, 32.3, 42.5, 57.6, 36.1, 32.8, 31.0, 65.5, 37.4),
        ("Gemini-2.5-Pro", 63.8, 44.3, 55.6, 57.6, 67.5, 55.1, 42.5, 44.9, 95.2, 57.9),
        ("LLaVA-OV 8B", 17.8, 6.9, 12.9, 15.7, 14.2, 13.0, 12.0, 12.7, 86.3, 19.3),
        ("UNIVID-7B", 56.3, 51.3, 50.2, 57.7, 50.1, 54.3, 32.4, 28.9, 82.3, 39.1),
        ("UNIVID-1B", 53.6, 49.1, 49.9, 55.3, 47.7, 52.1, 30.6, 27.4, 82.8, 37.5),
        ("w/o Hybrid Data", 37.9, 35.5, 33.5, 41.1, 30.4, 37.5, 18.4, 15.8, 81.8, 23.1),
        ("w/o Human Data", 29.1, 22.9, 18.9, 29.4, 23.9, 26.1, 15.8, 15.8, 82.2, 23.0),
    ]
    cols = (
        "model",
        "violence",
        "sex_abuse",
        "mental_health",
        "regulated_act",
        "integrity",
        "vio_rec",
        "non_vio_rec",
        "rec",
        "prec",
        "f1",
    )
    return [dict(zip(cols, row, strict=True)) for row in rows]


def table_iv_deployment_cost() -> list[dict[str, float | str | None]]:
    """Table 4 — VLM deployment cost per 1M videos."""
    return [
        {"model": "GPT-4.1", "vio_rec": 36.1, "cost_usd": 4830.0},
        {"model": "Gemini-2.5-Pro", "vio_rec": 55.1, "cost_usd": 3444.0},
        {"model": "LLaVA-OV", "vio_rec": 13.0, "cost_usd": None},
        {"model": "UNIVID", "vio_rec": 54.3, "cost_usd": 180.0},
    ]


def table_v_risk_filter_ablation() -> list[dict[str, float | str]]:
    """Table 5 — UNIVID embedding on risk filter + trend detector @65% precision."""
    return [
        {"embeddings": "w/o UNIVID", "vio_rec": 72.3, "leak_rec": 33.3, "trend_rec": 56.7},
        {"embeddings": "w/ UNIVID", "vio_rec": 78.2, "leak_rec": 59.8, "trend_rec": 86.7},
    ]


def table_vi_moderation_actor_ablation() -> list[dict[str, float | str]]:
    """Table 6 — UNIVID-Lite and RAG on moderation actor."""
    return [
        {"actor": "Recall", "vio_rate": 1.38, "vio_prec": 76.0, "leak_rec": 39.3},
        {"actor": "UNIVID-Lite", "vio_rate": 1.34, "vio_prec": 85.4, "leak_rec": 51.1},
        {"actor": "UNIVID-Lite + RAG", "vio_rate": 1.48, "vio_prec": 78.3, "leak_rec": 53.6},
    ]


def table_vii_capbench_domain_stats() -> list[dict[str, int | str]]:
    """Table 7 — CapBench domain distribution."""
    from ltx_trainer.univid.capbench import CAPBENCH_DOMAIN_STATS

    return list(CAPBENCH_DOMAIN_STATS)


def table_viii_architecture_ablation() -> list[dict[str, float | str]]:
    """Table 9 — UNIVID-7B architecture ablation on CapBench."""
    return [
        {
            "model": "UNIVID-7B",
            "violence": 56.3,
            "sex_abuse": 51.3,
            "mental_health": 50.2,
            "regulated_act": 57.7,
            "integrity": 32.4,
            "non_vio_rec": 28.9,
            "rec": 82.3,
            "prec": 39.1,
            "f1": 39.1,
        },
        {
            "model": "w/ EvaCLIP",
            "violence": 46.6,
            "sex_abuse": 44.2,
            "mental_health": 39.4,
            "regulated_act": 46.9,
            "integrity": 24.3,
            "non_vio_rec": 21.6,
            "rec": 77.1,
            "prec": 30.0,
            "f1": 30.0,
        },
        {
            "model": "w/ AnyRes",
            "violence": 56.7,
            "sex_abuse": 51.4,
            "mental_health": 48.9,
            "regulated_act": 58.4,
            "integrity": 32.5,
            "non_vio_rec": 28.7,
            "rec": 82.4,
            "prec": 38.7,
            "f1": 38.7,
        },
    ]


def table_ix_training_hyperparams() -> dict[str, str | float | int]:
    """Table 10 — UNIVID training hyperparameters."""
    return {
        "epochs": 2,
        "per_device_batch_size": 8,
        "gradient_accumulation": 2,
        "learning_rate": 1e-5,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
    }


def pipeline_demo(cfg: UnividConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    return {
        "risk_filter": run_risk_filter_smoke(seed=seed),
        "moderation_actor": run_lite_rag_smoke(seed=seed + 1),
        "prompts": prompt_bundle(),
        "univid_7b_vio_rec": next(r["vio_rec"] for r in table_iii_capbench_results() if r["model"] == "UNIVID-7B"),
    }


def evaluation_demo(cfg: UnividConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    smoke = evaluation_smoke(seed=0)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["evaluation_smoke"] = smoke
    demo["production_leakage_reduction_pct"] = (cfg or UnividConfig()).leakage_reduction_pct
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_training_stages": table_i_training_stages(),
        "table_ii_capbench_comparison": table_ii_capbench_comparison(),
        "table_iii_capbench_results": table_iii_capbench_results(),
        "table_iv_deployment_cost": table_iv_deployment_cost(),
        "table_v_risk_filter_ablation": table_v_risk_filter_ablation(),
        "table_vi_moderation_actor_ablation": table_vi_moderation_actor_ablation(),
        "table_vii_capbench_domain_stats": table_vii_capbench_domain_stats(),
        "table_viii_architecture_ablation": table_viii_architecture_ablation(),
        "table_ix_training_hyperparams": table_ix_training_hyperparams(),
    }
