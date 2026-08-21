"""Framework card, paper tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.figma.caption import caption_saturation_demo
from ltx_trainer.figma.config import FigmaConfig
from ltx_trainer.figma.loss import loss_demo


def framework_card(cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "project_url": c.project_url,
        "task": "Fine-Grained Music Retrieval",
        "method": "multi_view_contrastive (global + frame/token InfoNCE)",
        "encoders": {
            "audio": c.audio_encoder,
            "text": c.text_encoder,
            "frozen_params_m": c.frozen_params_m,
            "trainable_params_m": c.trainable_params_m,
        },
        "fgmcaps": {
            "train": c.fgmcaps_train,
            "val": c.fgmcaps_val,
            "test": c.fgmcaps_test,
            "attributes": list(c.fgmcaps_attributes),
            "extraction_tools": list(c.extraction_tools),
            "caption_llm": c.caption_llm,
        },
        "training": {
            "alpha": c.alpha_global,
            "temperature": c.temperature,
            "batch_size": c.batch_size,
            "epochs": c.epochs,
            "lr": c.learning_rate,
        },
        "headline": headline_results(c),
    }


def headline_results(cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    rel = (c.fmacaps_t2a_r1 - c.clamp3_fmacaps_t2a_r1) / c.clamp3_fmacaps_t2a_r1 * 100.0
    return {
        "musicbench_t2a_r1": c.musicbench_t2a_r1,
        "musicbench_a2t_r1": c.musicbench_a2t_r1,
        "fmacaps_t2a_r1": c.fmacaps_t2a_r1,
        "fgmcaps_test_t2a_r1": c.fgmcaps_test_t2a_r1,
        "relative_improvement_fmacaps_pct": round(rel, 1),
        "caption_saturation_tokens": c.caption_saturation_tokens,
    }


def table1_dataset_comparison() -> list[dict[str, Any]]:
    """Table 1 — dataset scale and music-theoretic caption attributes."""
    return [
        {
            "dataset": "JamendoMaxCaps",
            "train": 189_515,
            "test": 0,
            "chord": False,
            "tempo": False,
            "beat": False,
            "key": False,
        },
        {
            "dataset": "Music4All",
            "train": 108_042,
            "test": 0,
            "chord": False,
            "tempo": True,
            "beat": False,
            "key": True,
        },
        {
            "dataset": "MusicBench",
            "train": 52_768,
            "test": 400,
            "chord": True,
            "tempo": True,
            "beat": True,
            "key": True,
        },
        {
            "dataset": "FGMCaps",
            "train": 380_878,
            "test": 10_000,
            "chord": True,
            "tempo": True,
            "beat": True,
            "key": True,
        },
    ]


def table2_musicbench() -> list[dict[str, Any]]:
    """Table 2 — MusicBench retrieval R@K (selected models)."""
    return [
        {
            "model": "CLAMP3",
            "t2a_r1": 28.43,
            "t2a_r5": 57.87,
            "a2t_r1": 5.08,
            "a2t_r5": 24.37,
        },
        {
            "model": "MuQ-MuLaN",
            "t2a_r1": 20.81,
            "t2a_r5": 47.71,
            "a2t_r1": 17.76,
            "a2t_r5": 43.65,
        },
        {
            "model": "LAION-CLAP (FGMCaps continued)",
            "t2a_r1": 10.66,
            "t2a_r5": 36.55,
            "a2t_r1": 13.71,
            "a2t_r5": 36.55,
        },
        {
            "model": "FIGMA",
            "t2a_r1": 34.52,
            "t2a_r5": 65.99,
            "a2t_r1": 39.09,
            "a2t_r5": 68.02,
        },
    ]


def table3_fmacaps_eval() -> list[dict[str, Any]]:
    """Table 3 — FMACaps-Eval retrieval R@K."""
    return [
        {"model": "CLAMP3", "t2a_r1": 7.50, "t2a_r5": 20.70, "a2t_r1": 1.10, "a2t_r5": 4.10},
        {
            "model": "LAION-CLAP (FGMCaps continued)",
            "t2a_r1": 6.10,
            "t2a_r5": 18.30,
            "a2t_r1": 6.00,
            "a2t_r5": 20.00,
        },
        {"model": "FIGMA", "t2a_r1": 13.00, "t2a_r5": 28.00, "a2t_r1": 13.20, "a2t_r5": 33.30},
    ]


def table4_perturbation_a2t() -> list[dict[str, Any]]:
    """Table 4 — hard-negative attribute perturbation (A2T R@1)."""
    return [
        {"attribute_changed": "Original", "a2t_r1": 46.53, "a2t_r5": 74.97},
        {"attribute_changed": "Key", "a2t_r1": 38.90, "a2t_r5": 67.97},
        {"attribute_changed": "BPM", "a2t_r1": 40.30, "a2t_r5": 67.87},
        {"attribute_changed": "Tempo marking", "a2t_r1": 35.77, "a2t_r5": 65.87},
        {"attribute_changed": "Beat count", "a2t_r1": 34.87, "a2t_r5": 65.17},
        {"attribute_changed": "Chords", "a2t_r1": 43.20, "a2t_r5": 65.80},
    ]


def table5_fgmcaps_test() -> list[dict[str, Any]]:
    """Table 5 — FGMCaps test set (selected)."""
    return [
        {"model": "CLAMP3", "t2a_r1": 2.22, "a2t_r1": 0.35},
        {"model": "MuQ-MuLaN", "t2a_r1": 0.85, "a2t_r1": 1.47},
        {"model": "FIGMA", "t2a_r1": 26.15, "a2t_r1": 26.86},
    ]


def table8_source_splits() -> list[dict[str, Any]]:
    """Table 8 — FGMCaps source dataset splits."""
    return [
        {"dataset": "MTG-Jamendo", "train": 48_709, "val": 2_707, "test": 2_707},
        {"dataset": "Music4All", "train": 100_750, "val": 3_646, "test": 3_646},
        {"dataset": "JamendoMaxCaps", "train": 180_541, "val": 3_647, "test": 3_647},
        {"dataset": "MusicBench", "train": 50_878, "val": 0, "test": 0},
        {"dataset": "FGMCaps (Total)", "train": 380_878, "val": 10_000, "test": 10_000},
    ]


def experimental_protocol(cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    return {
        "clip_seconds": c.clip_seconds,
        "audio_frames": c.audio_frames,
        "text_tokens": c.text_tokens,
        "projection": "2-layer Transformer encoder + linear → 512-d",
        "metrics": ["R@1", "R@5", "R@10", "R@20"],
        "benchmarks": ["MusicBench", "FMACaps-Eval", "FGMCaps test"],
        "negative_set": "in-batch; ablation uses 8× batch size",
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_dataset_comparison": table1_dataset_comparison(),
        "table2_musicbench": table2_musicbench(),
        "table3_fmacaps_eval": table3_fmacaps_eval(),
        "table4_perturbation_a2t": table4_perturbation_a2t(),
        "table5_fgmcaps_test": table5_fgmcaps_test(),
        "table8_source_splits": table8_source_splits(),
    }


def evaluation_demo(seed: int = 0, cfg: FigmaConfig | None = None) -> dict[str, Any]:
    c = cfg or FigmaConfig()
    figma_mb = next(r for r in table2_musicbench() if r["model"] == "FIGMA")
    figma_fm = next(r for r in table3_fmacaps_eval() if r["model"] == "FIGMA")
    return {
        "loss": loss_demo(seed=seed, cfg=c),
        "caption_saturation": caption_saturation_demo(c),
        "musicbench_t2a_r1": figma_mb["t2a_r1"],
        "fmacaps_t2a_r1": figma_fm["t2a_r1"],
        "beats_clamp3_on_musicbench": figma_mb["t2a_r1"] > c.clamp3_t2a_r1,
    }


def pipeline_demo(seed: int = 42, cfg: FigmaConfig | None = None) -> dict[str, Any]:
    return {
        "framework": framework_card(cfg),
        "evaluation": evaluation_demo(seed=seed, cfg=cfg),
        "benchmarks": benchmarks_bundle(),
        "protocol": experimental_protocol(cfg),
    }
