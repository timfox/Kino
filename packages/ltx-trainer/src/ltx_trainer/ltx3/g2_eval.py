"""G2 acceptance: LongAV-Compass hooks after minute render."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ltx_trainer.ltx3.render_plan import MinuteRenderPlan, render_plan_longav_example
from ltx_trainer.ltx3.temporal_upscale import temporal_upscale_post_concat_command
from ltx_trainer.longav_compass.ltx_bridge import ltx_minute_av_eval_plan


def g2_post_render_smoke(
    *,
    out_dir: str | Path,
    work_root: str | None = None,
    include_temporal: bool = True,
) -> dict[str, Any]:
    """Ordered smoke checklist after minute compose (no GPU scoring)."""
    plan = render_plan_longav_example(out_dir=out_dir)
    eval_plan = ltx_minute_av_eval_plan(work_root=work_root or "")
    temporal = (
        temporal_upscale_post_concat_command(
            input_mp4=plan.full_output,
            output_mp4=str(Path(plan.full_output).with_name(Path(plan.full_output).stem + "_temporal_x2.mp4")),
        )
        if include_temporal
        else None
    )
    return {
        "gate": "G2_minute_longform",
        "render_plan": plan.to_dict(),
        "longav_eval_plan": eval_plan,
        "temporal_upscale": temporal,
        "smoke_commands": [
            "./scripts/kino-ltx3-minute-render.sh dry-run --longav",
            "./scripts/kino-ltx3-minute-render.sh run --longav",
            "./scripts/kino-ltx3-minute-render.sh temporal-run --longav",
            "pytest tests/test_longav_compass_tools.py -q",
            "python -m ltx_trainer.longav_compass eval-demo",
        ],
        "acceptance": [
            "concat full_video exists under out_dir",
            "event clips extractable via canonical_events_json",
            "LongAV-Compass eval-demo passes on bundled examples",
        ],
    }


def g2_validate_render_plan(plan: MinuteRenderPlan) -> list[str]:
    """Non-GPU validation issues beyond InferEnv checks."""
    issues: list(plan.validation_issues)
    if not plan.segments:
        issues.append("no segments in render plan")
    if plan.segments and not plan.concat_shell.strip():
        issues.append("missing concat shell")
    return issues
