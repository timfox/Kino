"""VOICEGIRAFFE framework card, demos, and benchmark bundles."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.eval import duration_decay_curve, eval_smoke, inference_mode_scores, run_eval
from ltx_trainer.voicegiraffe.hub_audio import hub_audio_smoke
from ltx_trainer.voicegiraffe.hub_loader import hub_loader_smoke
from ltx_trainer.voicegiraffe.hub_manifest import hub_manifest_smoke
from ltx_trainer.voicegiraffe.pool_eval import pool_eval_smoke, run_sampled_pool_eval
from ltx_trainer.voicegiraffe.layout import LIMITATIONS
from ltx_trainer.voicegiraffe.metrics import (
    memory_asymmetry,
    table_1_benchmark_comparison,
    table_2_leaderboard,
    table_3_lrm_ablation,
    table_4_language_bias,
)
from ltx_trainer.voicegiraffe.taxonomy import benchmark_stats, domain_coverage, task_taxonomy


def framework_card(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    return {
        "name": "VOICEGIRAFFE",
        "paper": cfg.paper_arxiv,
        "org": cfg.org,
        "idea": (
            "First bilingual hour-scale audio QA benchmark for LALMs: 1,500 MC items over "
            "123 native long-form recordings (~113 h) across five domains, with single-hop "
            "perception and multi-hop reasoning tiers."
        ),
        "stats": benchmark_stats(cfg),
        "domains": domain_coverage(),
        "task_taxonomy": task_taxonomy(),
        "inference_modes": ["E2E", "cascaded caption aggregation", "reasoning-enhanced cascading"],
        "human_reference_pct": cfg.human_overall_pct,
        "limitations": list(LIMITATIONS),
    }


def headline_results(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    mem = memory_asymmetry(cfg)
    return {
        "challenge": (
            f"Only {cfg.best_e2e_model} E2E ({cfg.best_e2e_overall_pct}%) beats human ({cfg.human_overall_pct}%); "
            f"best open-source cascade {cfg.best_opensource_cascade_pct}%"
        ),
        "inference_paradox": "E2E best for strong native long-context; cascade stabilizes weak models; LRM helps open-source but can bottleneck proprietary",
        "memory_bottleneck": (
            f"Models CA>ET; humans ET>CA by {mem['human_event_tracking_minus_causal_pct']:.1f}%"
        ),
        "opensource_lrm": f"Cascade {cfg.opensource_cascade_avg_pct}% → LRM {cfg.opensource_lrm_avg_pct}% (+{cfg.opensource_lrm_avg_pct - cfg.opensource_cascade_avg_pct:.1f}%)",
    }


def pipeline_demo(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    return {
        "n_qa": cfg.n_qa_total,
        "eval": eval_smoke(cfg, seed=seed),
        "cascade_eval": run_eval(seed=seed),
        "hub": hub_loader_smoke(),
        "hub_manifest": hub_manifest_smoke(cfg, seed=seed),
        "recording_registry": hub_loader_smoke()["matches_paper_scale"],
        "pool_eval": pool_eval_smoke(cfg, seed=seed),
        "hub_audio": hub_audio_smoke(cfg, seed=seed),
        "inference_modes": inference_mode_scores(cfg),
        "duration_decay": duration_decay_curve(),
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    return {
        "table_1_comparison": table_1_benchmark_comparison(),
        "table_2_leaderboard": table_2_leaderboard(cfg),
        "table_3_lrm_ablation": table_3_lrm_ablation(),
        "table_4_language_bias": table_4_language_bias(),
        "memory_asymmetry": memory_asymmetry(cfg),
        "task_taxonomy": task_taxonomy(),
        "benchmark_stats": benchmark_stats(cfg),
        "headline": headline_results(cfg),
    }
