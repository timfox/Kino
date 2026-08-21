"""Framework card, Table 1, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.usad2.config import Usad2Config
from ltx_trainer.usad2.distillation import distillation_demo


def framework_card(cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "collection_url": c.collection_url,
        "method": "domain_aware_layerwise_distillation + supervised_second_stage",
        "ssl_teachers": list(c.ssl_teachers),
        "supervised_teachers": list(c.supervised_teachers),
        "domains": list(c.domains),
        "domain_aware_alpha": c.domain_aware_alpha,
        "training": {
            "stage1_updates": c.stage1_updates,
            "stage2_updates": c.stage2_updates,
            "train_hours_ssl": c.train_hours_total,
        },
        "benchmarks": list(c.benchmarks),
        "headline": headline_results(c),
    }


def headline_results(cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    return {
        "xxlarge_plus_hear_avg": c.xxlarge_plus_hear_avg,
        "xxlarge_plus_marble_avg": c.xxlarge_plus_marble_avg,
        "xxlarge_plus_xares_track_a": c.xxlarge_plus_xares_track_a,
        "xxlarge_plus_xares_track_b": c.xxlarge_plus_xares_track_b,
        "xxlarge_params_m": c.xxlarge_params_m,
        "xxlarge_rtf_25hz": c.xxlarge_rtf_25hz,
    }


def table1_benchmark_averages() -> list[dict[str, Any]]:
    """Table 1 — HEAR / MARBLE / XARES-LLM encoder-only averages."""
    return [
        {
            "encoder": "SPEAR XLarge",
            "params_m": 600,
            "hear_avg": 82.6,
            "marble_avg": 75.1,
            "xares_track_a": 0.782,
            "xares_track_b": 0.457,
        },
        {
            "encoder": "Whisper Large",
            "params_m": 300,
            "hear_avg": 81.8,
            "marble_avg": 77.0,
            "xares_track_a": 0.691,
            "xares_track_b": 0.454,
        },
        {
            "encoder": "Multi-expert SSL (WavLM+ATST+MuQ)",
            "params_m": 734,
            "hear_avg": 82.0,
            "marble_avg": 76.1,
            "xares_track_a": 0.645,
            "xares_track_b": 0.462,
        },
        {
            "encoder": "Multi-expert Supervised (Whisper+AF3)",
            "params_m": 1274,
            "hear_avg": 81.8,
            "marble_avg": 72.4,
            "xares_track_a": 0.806,
            "xares_track_b": 0.685,
        },
        {
            "encoder": "USAD 2.0 XLarge",
            "params_m": 695,
            "hear_avg": 82.5,
            "marble_avg": 75.7,
            "xares_track_a": 0.708,
            "xares_track_b": 0.485,
            "stage": "ssl",
        },
        {
            "encoder": "USAD 2.0 XLarge+",
            "params_m": 695,
            "hear_avg": 84.4,
            "marble_avg": 75.0,
            "xares_track_a": 0.772,
            "xares_track_b": 0.611,
            "stage": "supervised",
        },
        {
            "encoder": "USAD 2.0 XXLarge+",
            "params_m": 1036,
            "hear_avg": 84.4,
            "marble_avg": 75.6,
            "xares_track_a": 0.783,
            "xares_track_b": 0.624,
            "stage": "supervised",
        },
    ]


def table2_ablation_small() -> list[dict[str, Any]]:
    """Table 2 — Small 25M ablation (PR / ESC-50 / NSynth)."""
    return [
        {"method": "USAD", "pr_per": 8.8, "esc50_acc": 80.3, "nsynth_acc": 55.1},
        {"method": "USAD 2.0", "pr_per": 8.7, "esc50_acc": 85.7, "nsynth_acc": 70.3},
        {"method": "w/o Domain-aware", "pr_per": 13.3, "esc50_acc": 83.4, "nsynth_acc": 69.1},
        {"method": "w/o Music Teacher", "pr_per": 8.5, "esc50_acc": 85.2, "nsynth_acc": 49.1},
        {"method": "w/o Music Data", "pr_per": 8.4, "esc50_acc": 84.3, "nsynth_acc": 53.2},
    ]


def table4_inference_efficiency() -> list[dict[str, Any]]:
    """Table 4 — RTF and peak GPU memory (30s audio, A5000)."""
    return [
        {"model": "USAD 2.0 Large", "params_m": 336, "framerate_hz": 50, "rtf": 0.0029, "gpu_gb": 1.2},
        {"model": "USAD 2.0 XLarge", "params_m": 695, "framerate_hz": 50, "rtf": 0.0051, "gpu_gb": 2.2},
        {"model": "USAD 2.0 XLarge", "params_m": 695, "framerate_hz": 25, "rtf": 0.0018, "gpu_gb": 1.7},
        {"model": "USAD 2.0 XXLarge", "params_m": 1036, "framerate_hz": 50, "rtf": 0.0077, "gpu_gb": 3.0},
        {"model": "USAD 2.0 XXLarge", "params_m": 1036, "framerate_hz": 25, "rtf": 0.0026, "gpu_gb": 2.4},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_benchmark_averages": table1_benchmark_averages(),
        "table2_ablation_small": table2_ablation_small(),
        "table4_inference_efficiency": table4_inference_efficiency(),
    }


def evaluation_demo(seed: int = 42, cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    distill = distillation_demo(seed=seed, cfg=c)
    xx = next(r for r in table1_benchmark_averages() if r["encoder"] == "USAD 2.0 XXLarge+")
    spear = next(r for r in table1_benchmark_averages() if r["encoder"] == "SPEAR XLarge")
    ablation = next(r for r in table2_ablation_small() if r["method"] == "USAD 2.0")
    wo_music = next(r for r in table2_ablation_small() if r["method"] == "w/o Music Teacher")
    return {
        "distillation": distill,
        "beats_spear_hear": xx["hear_avg"] > spear["hear_avg"],
        "beats_spear_xares_b": xx["xares_track_b"] > spear["xares_track_b"],
        "nsynth_gain_vs_usad_v1": ablation["nsynth_acc"] - c.usad_v1_nsynth_acc,
        "music_teacher_critical": ablation["nsynth_acc"] - wo_music["nsynth_acc"],
        "xxlarge_plus_track_b": xx["xares_track_b"],
    }


def pipeline_demo(seed: int = 42, cfg: Usad2Config | None = None) -> dict[str, Any]:
    c = cfg or Usad2Config()
    return {
        "framework": framework_card(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
        "benchmarks": benchmarks_bundle(),
    }
