"""DrawVideo framework card, paper tables, and smoke demos (arXiv:2605.23508)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.drawvideo.config import DrawVideoConfig, StoryboardShot, storyboard_from_shots
from ltx_trainer.drawvideo.layout import LIMITATIONS
from ltx_trainer.drawvideo.mock import toy_storyboard
from ltx_trainer.drawvideo.sketch import color_dodge_sketch, shot_boundary, toy_grayscale_line


def framework_card(cfg: DrawVideoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DrawVideoConfig()
    return {
        "name": "DrawVideo",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": (
            "Storyboard-driven long video: per shot (sketch Sk, appearance Ak, motion Mk), "
            "hierarchical global multi-shot / local single-sketch — sketch coloring → derivative keyframes → Wan first-last-frame clips."
        ),
        "pipeline_stages": [
            "Structured prompt decomposition (Qwen2.5)",
            "Sketch coloring (FLUX.1-dev + Canny-ControlNet)",
            "Derivative keyframes (FLUX.1 Kontext reference)",
            "First-last-frame video (Wan2.2-I2V)",
            "Shot concatenation → long video",
        ],
        "dataset": {
            "name": "SketchLongVideo",
            "triplets": cfg.sketchlongvideo_triplets,
            "sequences": cfg.sketchlongvideo_sequences,
            "appearance_words_avg": cfg.appearance_prompt_avg_words,
            "motion_words_avg": cfg.motion_prompt_avg_words,
        },
        "backends": {
            "coloring": cfg.sketch_coloring,
            "keyframes": cfg.derivative_keyframes,
            "video": cfg.video_backbone,
        },
    }


def table_quantitative_main() -> dict[str, dict[str, float]]:
    """Table 1 excerpt — DrawVideo vs baselines (paper values)."""
    return {
        "DrawVideo": {
            "LPIPS": 0.4751,
            "CLIP": 0.7806,
            "Edge_F1": 0.4664,
            "Temp_CLIP": 0.9225,
            "Temp_LPIPS": 0.3830,
            "Static_Align": 0.3781,
            "Story_Align": 0.3111,
            "Event_Comp": 0.7778,
            "Dyn_Control": 0.6973,
            "Dyn_Progress": 0.2493,
        },
        "SketchVideo_1kf": {
            "LPIPS": 0.6888,
            "CLIP": 0.7640,
            "Edge_F1": 0.3255,
            "Temp_CLIP": 0.8360,
            "Temp_LPIPS": 0.5683,
            "Static_Align": 0.3285,
            "Story_Align": 0.2923,
            "Event_Comp": 0.4444,
            "Dyn_Control": 0.5551,
            "Dyn_Progress": 0.0304,
        },
        "Wan2.2_prompt_only": {
            "LPIPS": float("nan"),
            "CLIP": float("nan"),
            "Edge_F1": float("nan"),
            "Temp_CLIP": 0.8788,
            "Temp_LPIPS": 0.3651,
            "Static_Align": 0.2580,
            "Story_Align": 0.2441,
            "Event_Comp": 0.3333,
            "Dyn_Control": 0.5070,
            "Dyn_Progress": 0.2014,
        },
        "Wan2.2_sketch_prompt": {
            "LPIPS": 0.4859,
            "CLIP": 0.7848,
            "Edge_F1": 0.3541,
            "Temp_CLIP": 0.8796,
            "Temp_LPIPS": 0.4893,
            "Static_Align": 0.3520,
            "Story_Align": 0.3023,
            "Event_Comp": 0.6667,
            "Dyn_Control": 0.6542,
            "Dyn_Progress": 0.1893,
        },
    }


def table_human_mos() -> dict[str, dict[str, float]]:
    """Table 5 excerpt — human evaluation MOS (1–5)."""
    return {
        "DrawVideo": {
            "Structural": 3.69,
            "Appearance": 3.72,
            "Motion": 3.56,
            "Controllability": 3.79,
            "Overall": 3.69,
        },
        "SketchVideo_1kf": {"Structural": 2.37, "Appearance": 2.51, "Motion": 2.41, "Controllability": 2.35, "Overall": 2.41},
        "Wan2.2_sketch_prompt": {"Structural": 3.42, "Appearance": 3.31, "Motion": 3.63, "Controllability": 3.03, "Overall": 3.35},
    }


def table_ablation_wan() -> dict[str, str]:
    """Sec. 5.2 — staged design vs Wan2.2 ablations."""
    return {
        "prompt_only": "Text-only Wan2.2; no sketch or intermediate keyframes.",
        "sketch_prompt": "Sketch + prompt directly to Wan2.2; no coloring or derivative expansion.",
        "DrawVideo": "Sketch → colored anchor → Kontext derivatives → first-last Wan clips.",
    }


def sketchlongvideo_stats() -> dict[str, Any]:
    """Sec. 3.4 dataset statistics."""
    cfg = DrawVideoConfig()
    return {
        "triplets_total": cfg.sketchlongvideo_triplets,
        "sequences": cfg.sketchlongvideo_sequences,
        "subset_triplets": {"self_collected": 201, "animeshooter": 932, "ai_generated": 100},
        "avg_keyframes_per_sequence": 9.79,
        "median_keyframes_per_sequence": 10,
        "dominant_sketch_resolution": "600×338",
    }


def pipeline_demo(cfg: DrawVideoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DrawVideoConfig()
    g, eroded = toy_grayscale_line()
    sketch = color_dodge_sketch(g, eroded)
    board = storyboard_from_shots(toy_storyboard())
    return {
        "shot_boundary_detected": shot_boundary(30.0, threshold=cfg.scene_detect_threshold),
        "sketch_pixels": len(sketch),
        "storyboard_shot_count": board["count"],
        "formulas": {
            "shot": "Shotk = Concat(V^1_k, ..., V^n_k)",
            "long": "V_long = Concat(Shot1, ..., Shotm)",
            "keyframe": "I^i_k = K(I^0_k, C^i_k)",
        },
    }


def evaluation_demo(cfg: DrawVideoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DrawVideoConfig()
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "paper_tables": {
            "quantitative": table_quantitative_main(),
            "human_mos": table_human_mos(),
            "ablation": table_ablation_wan(),
            "sketchlongvideo": sketchlongvideo_stats(),
        },
    }
