"""LTX 3.0-style evolution ladder and minute-generation workflow (Gopex fork)."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.longav_compass.ltx_bridge import ltx_minute_av_eval_plan
from ltx_trainer.ltx3.innovation_registry import (
    evolve_innovation_summary,
    innovation_stack,
    paper_coverage,
)
from ltx_trainer.ltx3.minute_compose import compose_minute_plan, segment_from_events


def framework_card() -> dict[str, Any]:
    return {
        "name": "Gopex LTX 3.0 vision",
        "base_model": "Lightricks LTX-2.3-22b (frozen DiT)",
        "text_stack": "Gemma 4 via Gopex native integration (not stock LTX pairing)",
        "stock_ltx_text": "Gemma 3 geometry at upstream release",
        "fork_adds": [
            "flat_dim bridge + fold → ltx-*-gemma4-native",
            "LTX_GEMMA_ENCODE_CAP long prompts",
            "AV-fold train weights + audio_encode align",
            "HDR LogC3 ladder",
            "minute_compose segment planner (composed 60s+ AV)",
            "innovation_registry — paper → pillar → hook → gate (ltx_trainer.ltx3)",
        ],
        "doc": "documents/LTX_3_VISION.md",
        "acceptance_gates": ["G1 delivery quality", "G2 minute composed T2AV", "G3 HDR", "G4 catalog+retrain"],
        "innovations": evolve_innovation_summary(),
    }


def evolve_ladder(*, work_root: str | None = None) -> dict[str, Any]:
    wr = work_root or os.environ.get("WORK_ROOT", "/run/media/tim/Expansion/gopex-ltx/gemma31b-r512")
    return {
        "framework": framework_card(),
        "work_root": wr,
        "phases": [
            {
                "id": "now",
                "summary": "Finish Phase 2 @ 12k; pd-horror split/prep/merge; re-encode audio_latents",
                "commands": [
                    "./scripts/kino-pd-horror-after-train.sh",
                    "GOPEX_AUDIO_ALIGN_VIDEO=1 ./scripts/kino-gemma4-native-path.sh prep-to-train",
                ],
            },
            {
                "id": "A_sdr_evolve",
                "summary": "Connector evolve + delivery + eval loop",
                "commands": [
                    "./scripts/kino-native-evolve.sh train-connectors",
                    "./scripts/kino-native-evolve.sh delivery",
                    "./scripts/kino-native-evolve.sh eval",
                ],
            },
            {
                "id": "B_minute_compose",
                "summary": "Composed minute AV (hero carry + concat); LongAV eval",
                "commands": [
                    "./scripts/kino-ltx3-evolve.sh minute-plan",
                    "./scripts/kino-ltx3-minute-render.sh render-plan --longav",
                    "./scripts/kino-ltx3-minute-render.sh dry-run --longav",
                    "./scripts/gopex-longav-compass.sh ltx-plan",
                ],
            },
            {
                "id": "C_hdr",
                "summary": "HDR preprocess → merge_native_hdr → train-hdr",
                "commands": ["./scripts/kino-native-evolve.sh hdr-ladder"],
            },
            {
                "id": "D_temporal_upscale",
                "summary": "Temporal x2 latent upscaler post-concat (minute path)",
                "status": "wired",
                "commands": [
                    "./scripts/kino-ltx3-minute-render.sh temporal-status",
                    "./scripts/kino-ltx3-minute-render.sh temporal-run --longav",
                    "python pipeline/temporal_upscale_kino.py --input full.mp4 --output full_temporal_x2.mp4",
                ],
                "doc": "documents/LTX_3_VISION.md",
            },
            {
                "id": "E_spatial_audio",
                "summary": "SwanSphere/Foley-Omni beyond mel proxy VAE",
                "status": "research",
                "stack": "ltx3_audio",
            },
            {
                "id": "F_innovation_registry",
                "summary": "Apply curated research stack to prep + train + eval",
                "commands": [
                    "./scripts/kino-ltx3-evolve.sh innovations",
                    "./scripts/kino-ltx3-evolve.sh stack ltx3_full",
                    "eval \"$(python tools/ltx_research_profile.py env ltx3_quality)\"",
                ],
            },
        ],
        "paper_coverage": paper_coverage(),
        "default_stack": innovation_stack("ltx3_full"),
    }


def minute_generation_plan(
    *,
    prompt: str,
    target_duration_s: float = 60.0,
    fps: float = 24.0,
    use_longav_example: bool = False,
) -> dict[str, Any]:
    """Merge segment planner output with LongAV eval workflow."""
    if use_longav_example:
        from ltx_trainer.longav_compass import example_t2av_performance_ads_l4

        plan = segment_from_events(example_t2av_performance_ads_l4(), fps=fps)
    else:
        plan = compose_minute_plan(
            global_prompt=prompt,
            target_duration_s=target_duration_s,
            fps=fps,
        )
    eval_plan = ltx_minute_av_eval_plan(target_duration_s=int(target_duration_s))
    return {
        "compose": plan.to_dict(),
        "inference_notes": [
            "Segment 0: inference.py or two_stage_hq_kino.py (GOPEX_LTX3_HQ=1).",
            "Segment n>0: --condition-image hero from previous segment last frame.",
            "Weak spans: RetakePipeline when upstream retake.py lands (manual fix for now).",
            "Concat: ./scripts/kino-ltx3-minute-render.sh run",
        ],
        "render": "./scripts/kino-ltx3-minute-render.sh render-plan --longav"
        if use_longav_example
        else "./scripts/kino-ltx3-minute-render.sh render-plan --prompt ...",
        "eval": eval_plan,
        "env": {
            "LTX_GEMMA_ENCODE_CAP": os.environ.get("LTX_GEMMA_ENCODE_CAP", "8192"),
            "NATIVE_LTX": os.environ.get("NATIVE_LTX", ""),
            "GOPEX_NATIVE_PHASE2_RUN": os.environ.get("GOPEX_NATIVE_PHASE2_RUN", ""),
        },
    }
