"""LTX / GOPEX I2V bridge notes for LiveSVG target generation (Sec. 3.3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livesvg.config import LiveSVGConfig
from ltx_trainer.livesvg.prompts import gemini_stage1_rubric, i2v_motion_prompt


def ltx_integration_notes(cfg: LiveSVGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiveSVGConfig()
    return {
        "stage": "target_video_generation",
        "recommended_backends": list(cfg.reference_generators),
        "gopex_path": (
            "Render recolored SVG → PNG condition frame → "
            "ltx_trainer.validation_sampler.GenerationConfig(condition_image=..., num_frames=keyframes)"
        ),
        "prompt_template": "livesvg.prompts.i2v_motion_prompt",
        "filtering": "Gemini two-stage scorer (10–20 seeds) before DiffVG fitting",
        "does_not": "Does not run DiffVG or SVG export inside LTX trainer — fitting is a separate CPU/GPU stage",
        "num_frames": cfg.num_keyframes,
        "resolution_note": f"Optimize at {cfg.opt_resolution}px; export previews at {cfg.export_pixels}px",
    }


def build_ltx_i2v_kwargs(
    motion_prompt: str,
    *,
    cfg: LiveSVGConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Keyword bundle aligned with ``validation_sampler.GenerationConfig`` field names."""
    cfg = cfg or LiveSVGConfig()
    full_prompt = i2v_motion_prompt(motion_prompt)
    return {
        "prompt": full_prompt,
        "negative_prompt": "3d rotation, color grading, gradients, shadows, new objects, camera orbit",
        "num_frames": cfg.num_keyframes,
        "frame_rate": 24.0,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "seed": seed,
        "generate_audio": False,
        "condition_image": "<recolored SVG raster tensor [3,H,W] in [0,1]>",
        "height": cfg.opt_resolution,
        "width": cfg.opt_resolution,
        "gemini_filter_rubric": gemini_stage1_rubric(),
    }
