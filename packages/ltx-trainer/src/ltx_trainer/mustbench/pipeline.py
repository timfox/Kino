"""MUSTBENCH framework card and paper tables (arXiv:2605.29300)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mustbench.config import MustBenchConfig
from ltx_trainer.mustbench.layout import LIMITATIONS
from ltx_trainer.mustbench.eval_suite import eval_suite_smoke
from ltx_trainer.mustbench.grpo_trainer import grpo_trainer_smoke, run_grpo_epochs
from ltx_trainer.mustbench.metrics import metrics_smoke
from ltx_trainer.mustbench.must_encoder import must_encoder_smoke
from ltx_trainer.mustbench.must_train import must_train_smoke, stage_pipeline
from ltx_trainer.mustbench.rewards import grpo_smoke
from ltx_trainer.mustbench.tasks import split_statistics, task_registry


def framework_card(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    return {
        "name": "MUSTBENCH",
        "paper": cfg.paper_arxiv,
        "idea": (
            "Music-expert-validated benchmark for temporal grounding in LALMs via five "
            "temporally grounded QA tasks. MUST is a four-stage recipe: MERT encoder "
            "adaptation, timestamped caption pretraining, QA SFT, and GRPO."
        ),
        "tasks": task_registry(cfg),
        "benchmark_stats": {
            "total_qa_pairs": cfg.total_qa_pairs,
            "unique_songs": cfg.unique_songs,
            "avg_duration": cfg.avg_duration,
        },
        "must_stages": stage_pipeline(cfg),
        "backbone": cfg.backbone,
        "limitations": list(LIMITATIONS),
    }


def table_i_benchmark_stats(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    return {
        "total_qa_pairs": cfg.total_qa_pairs,
        "unique_songs": cfg.unique_songs,
        "avg_duration": cfg.avg_duration,
        "per_task_test_n": {
            "TSG": cfg.tsg_n,
            "LTR": cfg.ltr_n,
            "TAD": cfg.tad_n,
            "GTO": cfg.gto_n,
            "MTR": cfg.mtr_n,
        },
    }


def table_iii_main_results() -> list[dict[str, Any]]:
    """Table 3 excerpt — selected models (Total Avg.)."""
    return [
        {"model": "Gemini 2.5 Flash", "total": 41.8, "TSG_onset": 60.0, "TSG_offset": 71.5, "GTO": 42.4},
        {"model": "Music Flamingo 8B", "total": 31.1, "TSG_onset": 53.0, "TSG_offset": 24.0, "GTO": 41.4},
        {"model": "Qwen 2.5 Omni 7B", "total": 24.3, "TSG_onset": 39.0, "TSG_offset": 3.5, "GTO": 46.5},
        {"model": "Qwen 3 Omni 30B", "total": 30.2, "TSG_onset": 62.5, "TSG_offset": 9.5, "GTO": 63.6},
        {"model": "MUST 3B", "total": 38.1, "TSG_onset": 35.5, "TSG_offset": 41.0, "GTO": 57.1},
        {"model": "MUST 7B", "total": 44.1, "TSG_onset": 55.5, "TSG_offset": 62.5, "GTO": 67.2},
    ]


def table_iv_ablation() -> list[dict[str, Any]]:
    """Table 4 — training stage ablation (7B, Total Avg.)."""
    return [
        {"setting": "Base model", "caption": False, "qa": False, "grpo": False, "total": 24.3},
        {"setting": "Caption only", "caption": True, "qa": False, "grpo": False, "total": 25.2},
        {"setting": "QA only", "caption": False, "qa": True, "grpo": False, "total": 36.8},
        {"setting": "Caption + QA", "caption": True, "qa": True, "grpo": False, "total": 41.9},
        {"setting": "Full (MUST)", "caption": True, "qa": True, "grpo": True, "total": 44.1},
        {"setting": "0 MUST tokens", "caption": True, "qa": True, "grpo": False, "total": 24.6},
        {"setting": "Uniform sampling", "caption": True, "qa": True, "grpo": False, "total": 39.8},
        {"setting": "Dynamic sampling", "caption": True, "qa": True, "grpo": False, "total": 41.9},
    ]


def headline_results(cfg: MustBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    return {
        "best_open_source": f"MUST 7B Total {cfg.must_7b_total} (+19.8 pp vs Qwen2.5-Omni 7B)",
        "offset_gain": f"MUST 7B offset Hit@3s {cfg.must_7b_tsg_offset_hit3}% (+59.0 pp vs base)",
        "vs_gemini": f"MUST 7B {cfg.must_7b_total} beats Gemini 2.5 Flash {cfg.gemini_25_flash_total}",
        "key_finding": "LALMs recognize transitions but fail precise temporal boundary estimation",
        "ablation_driver": "QA fine-tuning essential; GRPO improves temporal validity",
    }


def pipeline_demo(cfg: MustBenchConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or MustBenchConfig()
    train = must_train_smoke(seed=seed)
    metrics = metrics_smoke(seed=seed)
    grpo = grpo_smoke(cfg)
    encoder = must_encoder_smoke(cfg)
    eval_out = eval_suite_smoke()
    grpo_train = grpo_trainer_smoke(cfg)
    grpo_epochs = run_grpo_epochs(steps=6, cfg=cfg, seed=seed)
    return {
        "num_tasks": len(task_registry(cfg)),
        "train_qa_total": cfg.train_qa_total,
        "must_stages": train["num_stages"],
        "tsg_hit3_proxy": metrics["tsg_hit3"],
        "grpo_ordering_ok": grpo["ordering"],
        "must_7b_total_anchor": cfg.must_7b_total,
        "must_tokens": encoder["n_tokens"],
        "computed_tsg_hit3": eval_out["tsg_hit3"],
        "computed_gto_acc": eval_out["gto_acc"],
        "grpo_mean_reward": grpo_train["mean_reward"],
        "grpo_loss_finite": grpo_train["grpo_loss_finite"],
        "grpo_epochs": grpo_epochs["steps"],
        "grpo_reward_improved": grpo_epochs["reward_improved"],
    }


def evaluation_demo(*, seed: int = 42) -> dict[str, Any]:
    return {
        "headline": headline_results(),
        "demo": pipeline_demo(seed=seed),
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "tasks": task_registry(),
        "split_statistics": split_statistics(),
        "table_i_stats": table_i_benchmark_stats(),
        "table_iii_main_results": table_iii_main_results(),
        "table_iv_ablation": table_iv_ablation(),
        "must_stages": stage_pipeline(),
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
