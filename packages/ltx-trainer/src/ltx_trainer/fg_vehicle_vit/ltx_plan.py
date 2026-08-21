"""Gopex hooks for roadside overtaking video + cyclist safety analytics."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig


def ltx_integration_plan(cfg: FgVehicleVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgVehicleVitConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_case": (
            "Annotate overtaking vehicle body type on bicycle-lane / naturalistic video "
            "for injury-risk exposure (SUV vs passenger car vs commercial truck)."
        ),
        "gopex_workflow": [
            "Decode corridor video @ 10 FPS (ffmpeg) → frame folders",
            "Run fg_vehicle_vit inference (RT-DETR + ViT) per frame or event",
            "Join predictions to frame-indexed ground-truth CSV (overtaking events)",
            "Export counts by body type for safety / policy studies",
            "Optional: attach labels to LTX caption sidecars for conditioned generation",
        ],
        "env_knobs": {
            "GOPEX_FG_VEHICLE_VIT_CONF_S1": str(cfg.stage1_conf_threshold),
            "GOPEX_FG_VEHICLE_VIT_ABSTAIN": str(cfg.abstention_threshold),
            "GOPEX_FG_VEHICLE_VIT_WEIGHTS": "Path to fine-tuned ViT checkpoint",
            "GOPEX_FG_VEHICLE_VIT_RT_DETR": cfg.rtdetr_checkpoint,
        },
        "related_datasets": [
            "Ann Arbor N. Division annotations (paper repo)",
            cfg.open_data_url,
        ],
        "not_in_scope": "Full HuggingFace RT-DETR + ViT weight download in CI smoke",
    }
