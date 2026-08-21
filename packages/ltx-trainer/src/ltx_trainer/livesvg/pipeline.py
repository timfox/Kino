"""LiveSVG framework card, demos, and benchmark manifest."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livesvg.challengesvg import challengesvg_manifest, example_records_preview
from ltx_trainer.livesvg.config import BASELINES, LiveSVGConfig
from ltx_trainer.livesvg.experiments import run_full_experiment, run_synthetic_fitting_experiment
from ltx_trainer.livesvg.paper_report import compare_experiment_to_paper
from ltx_trainer.livesvg.fitting import pipeline_stage_summary, progressive_activation_schedule, toy_synthetic_fitting_step
from ltx_trainer.livesvg.ltx_bridge import build_ltx_i2v_kwargs, ltx_integration_notes
from ltx_trainer.livesvg.paper_tables import (
    human_preference_rates,
    table_ablation_gemini,
    table_aniclipart_quantitative,
    table_benchmark_structure,
    table_challengesvg_quantitative,
    table_runtime_minutes,
)
from ltx_trainer.livesvg.prompts import gemini_stage1_rubric, i2v_motion_prompt, method_capability_table
from ltx_trainer.livesvg.recolor import recolorization_report


def framework_card(cfg: LiveSVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiveSVGConfig()
    return {
        "name": "LiveSVG",
        "paper": cfg.paper_arxiv,
        "title": "Zero-Shot SVG Animation via Video Generation",
        "project_page": cfg.project_page,
        "paradigm": "Decouple motion (frozen I2V target mp4) from vector fitting (DiffVG)",
        "motion_representation": {
            "global": "per-group 8-DOF homography per keyframe",
            "local": "per-path Bézier control-point offsets",
        },
        "preprocessing": [
            "LLM semantic grouping",
            "sphere-packing recolorization",
            "optional SAM2 layer reorder",
        ],
        "target_generators": list(cfg.reference_generators),
        "default_generator": cfg.default_reference_generator,
        "optimization": {
            "keyframes": cfg.num_keyframes,
            "resolution": cfg.opt_resolution,
            "iterations": cfg.opt_iterations,
            "renderer": cfg.renderer,
        },
        "baselines": list(BASELINES),
        "benchmarks": {
            "AniClipart": cfg.aniclipart_examples,
            "ChallengeSVG": cfg.challengesvg_examples,
        },
        "capabilities": method_capability_table()["LiveSVG"],
    }


def paper_limitations() -> list[str]:
    return [
        "Target I2V quality and SVG compatibility (color drift, invented parts, 3D rotation) bound fitting.",
        "DiffVG, Veo/LTX/WAN, Gemini, SAM2, TAPNext, and RMBG are external — not bundled.",
        "Heavy occlusion can still break fixed-topology MSE tracking.",
        "Recolorization is supervision-only; final export restores original path colors.",
    ]


def benchmark_manifest(cfg: LiveSVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiveSVGConfig()
    return {
        "benchmarks": {
            "AniClipart": {"n": cfg.aniclipart_examples, "metrics": list(table_aniclipart_quantitative()["LiveSVG (Veo 3.1)"].keys())},
            "ChallengeSVG": challengesvg_manifest(),
        },
        "structure_stats": table_benchmark_structure(),
        "human_preferences": human_preference_rates(),
        "runtime_minutes": table_runtime_minutes(),
        "ablations": table_ablation_gemini(),
    }


def evaluation_demo(cfg: LiveSVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiveSVGConfig()
    motion = "a character waves hello"
    schedule = progressive_activation_schedule(
        min(cfg.opt_iterations, 500),
        cfg.num_keyframes,
        interval=cfg.progressive_interval_iters,
    )
    return {
        "i2v_prompt": i2v_motion_prompt(motion),
        "gemini_rubric": gemini_stage1_rubric(),
        "recolor": recolorization_report(24, seed=3),
        "progressive_schedule_tail": schedule[-5:],
        "fitting_losses": toy_synthetic_fitting_step(cfg=cfg),
        "multiframe_experiment": run_synthetic_fitting_experiment(cfg=cfg, opt_steps=120),
        "full_experiment": run_full_experiment(cfg=cfg, opt_steps=120),
        "paper_report": compare_experiment_to_paper(
            run_synthetic_fitting_experiment(cfg=cfg, opt_steps=120)
        ),
        "ltx_i2v": build_ltx_i2v_kwargs(motion, cfg=cfg),
        "ltx_notes": ltx_integration_notes(cfg),
        "pipeline": pipeline_stage_summary(),
        "challengesvg_preview": example_records_preview(3),
        "paper_best_xclip_aniclipart": table_aniclipart_quantitative()["LiveSVG (Veo 3.1)"]["XCLIP"],
    }
