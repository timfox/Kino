"""Sphere360 pipeline demos (clip parse + LTX plan, no Hub download)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sphere360._bootstrap import ensure_repo_root
from ltx_trainer.sphere360.paper import framework_card


def pipeline_demo(*, project: str = "sphere360-hdr-timelapse") -> dict[str, Any]:
    ensure_repo_root()
    from gopex_datasets.sphere360 import parse_clip_id
    from gopex_datasets.sphere360.ltx_bridge import hdr_timelapse_lora_plan

    ref = parse_clip_id("dQw4w9WgXcQ_012")
    plan = hdr_timelapse_lora_plan(project_name=project)
    return {
        "framework": framework_card(),
        "clip_id": ref.clip_id,
        "youtube_id": ref.youtube_video_id,
        "segment_index": ref.segment_index,
        "plan_phases": len(plan["steps"]),
        "clip_duration_s": plan["clip_duration_s"],
    }


def evaluation_demo() -> dict[str, Any]:
    ensure_repo_root()
    from gopex_datasets.sphere360.knowledge import sphere360_knowledge_blob

    kb = sphere360_knowledge_blob()
    demo = pipeline_demo()
    return {
        "hub_dataset": kb["hub_dataset"],
        "train_clips": kb["scale"]["train_clips"],
        "demo_clip_parsed": demo["clip_id"],
        "plan_phases": demo["plan_phases"],
    }
