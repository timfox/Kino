"""LTX 2.3 minute-scale audio-visual evaluation plan (GOPEX hook)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.longav_compass.annotation import BenchmarkCase, build_generation_prompt
from ltx_trainer.longav_compass.config import LongAVCompassConfig
from ltx_trainer.longav_compass.segmentation import canonical_events_json, event_segment_specs


def ltx_minute_av_eval_plan(
    *,
    project_name: str = "longav-compass-ltx",
    work_root: str = "/run/media/tim/Expansion/gopex-ltx/gemma31b-r512",
    ltx_config: str = "configs/ltx2_av_lora_gemma4_31b_hdr_logc3_connectors.yaml",
    target_duration_s: int = 60,
) -> dict[str, Any]:
    """Ordered steps to generate and score minute-long AV clips with LTX under LongAV-Compass."""
    cfg = LongAVCompassConfig()
    out_root = f"{work_root.rstrip('/')}/eval/{project_name}"
    return {
        "goal": "Evaluate LTX 2.3 (or LoRA) on LongAV-Compass T2AV/I2AV/V2AV diagnostic protocol",
        "benchmark": {
            "paper": cfg.paper_arxiv,
            "samples": cfg.n_samples,
            "duration_s": [cfg.target_duration_s_min, cfg.target_duration_s_max],
        },
        "related_gopex_evaluators": [
            "ltx_trainer.longav_compass (minute-scale X2AV; this plan)",
            "ltx_trainer.avbench (short T2AV human-aligned ten-metric suite)",
            "gopex_datasets.openvid_1m / seedance2_prompts (prompt mining, not minute eval)",
        ],
        "prerequisites": [
            "LongAV-Compass annotation JSON from upstream release (or GOPEX example cases)",
            "LTX-2.3 AV weights + Gemma text encoder (documents/LTX_WEIGHTS.md)",
            "FFmpeg for event/boundary clip extraction",
        ],
        "steps": [
            {
                "phase": "1_prompts",
                "summary": "Convert each case to native LTX prompts (global + optional per-event).",
                "commands": [
                    "PYTHONPATH=kino/packages/ltx-trainer/src "
                    "python -m ltx_trainer.longav_compass prompt t2av",
                    "# For each case: build_generation_prompt(case) → video_prompt / audio_prompt",
                ],
            },
            {
                "phase": "2_generate",
                "summary": f"Generate ≥{target_duration_s}s outputs (native multi-segment or autoregressive).",
                "notes": [
                    "Preserve event order from annotation; avoid fixed-window chunking for scoring.",
                    "I2AV: pass reference image via model conditioning API.",
                    "V2AV: 10–15s reference clip + continuation script for remaining ~45–50s.",
                ],
                "output_layout": f"{out_root}/<model_name>/<case_id>/full_video.mp4",
            },
            {
                "phase": "3_segment",
                "summary": "Extract event clips and 2s boundary windows (canonical_events.json).",
                "commands": [
                    "# Use canonical_events_json(case) for ffmpeg -ss/-to per event",
                    f"# Write under {out_root}/<model>/<case_id>/events/ and .../boundaries/",
                ],
            },
            {
                "phase": "4_score",
                "summary": "Run LongAV-Compass metrics (VQA, VQ, Cont., Trans., Hol., TVAlign, audio).",
                "commands": [
                    "longav_compass_eval_demo  # smoke on bundled examples",
                    "longav_compass_benchmarks  # compare to Tables 3–7 excerpts",
                    "# Full run: Gemini 3.1 Pro judge + CLIP/DINO when upstream scripts release",
                ],
                "primary_metrics": ["VQA", "VQ", "Cont.", "Trans.", "Hol.", "TVAlign", "AVS", "AudQ", "AudL"],
                "i2av_extra": ["IV1", "ImgAlign"],
            },
            {
                "phase": "5_report",
                "summary": "Task-specific leaderboards + scenario/difficulty slices.",
                "notes": [
                    "Performance Ads is the most discriminative scenario (paper Sec. 4.4).",
                    "Compare against Seedance 2.0 / Kling 3.0 table excerpts via longav_compass_benchmarks.",
                ],
            },
        ],
        "ltx_config_hint": ltx_config,
    }


def plan_for_case(case: BenchmarkCase, *, model_name: str = "LTX 2.3") -> dict[str, Any]:
    """Per-case generation + segmentation checklist."""
    return {
        "case_id": case.case_id,
        "task": case.task,
        "model": model_name,
        "generation": build_generation_prompt(case),
        "artifacts": canonical_events_json(case),
        "n_events": len(event_segment_specs(case)),
    }
