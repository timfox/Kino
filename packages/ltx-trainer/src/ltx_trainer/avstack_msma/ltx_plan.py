"""Gopex / LTX hooks for simulator-generated multi-view autonomy data."""

from __future__ import annotations

from typing import Any

from ltx_trainer.avstack_msma.config import AvstackMsmaConfig


def ltx_integration_plan(cfg: AvstackMsmaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AvstackMsmaConfig()
    return {
        "paper": cfg.paper_arxiv,
        "role_in_gopex": (
            "Synthetic MS/MA logs feed caption+prep for multi-camera ERP/T2V and VLA-style driving research "
            "(LLavida-class assistants, §4.4)."
        ),
        "recommended_exports": [
            "Synchronized RGB (+ optional depth/semantic) per agent → clip folders",
            "Per-sensor COCO boxes → grounding prompts / detection QA",
            "Infrastructure + aerial viewpoints → domain-shift eval splits",
        ],
        "prep_workflow": [
            "carla-sandbox run → AVstack log_root",
            "postprocess labels per sensor frame",
            "COCO or native export → GOPEX dataset manifest",
            "kino-parallel-prep captions on RGB streams",
            "Optional merge into merged_native for LTX fine-tune",
        ],
        "env_knobs": {
            "GOPEX_AVSTACK_MSMA_LOG_ROOT": "Path to generated run folder",
            "GOPEX_AVSTACK_MSMA_DOMAIN": "ground | aerial | infrastructure",
            "GOPEX_AVSTACK_MSMA_SEED": "CARLA random seed for reproducible regeneration",
            "GOPEX_AVSTACK_MSMA_DURATION_S": "Collection duration per scenario YAML",
        },
        "security_trust": "ROS2 multi-agent testbed + trust fusion studies (§4.3) — use generated logs as repeatable attack surfaces",
        "not_in_scope": "Full CARLA binary install and Unreal runtime in CI smoke",
    }
