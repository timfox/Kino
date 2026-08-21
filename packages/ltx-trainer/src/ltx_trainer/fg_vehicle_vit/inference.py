"""Two-stage pipeline decision logic (Fig. 1, §2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ltx_trainer.fg_vehicle_vit.abstention import UNKNOWN_LABEL, classify_with_abstention
from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig
from ltx_trainer.fg_vehicle_vit.stage1 import Detection, filter_detections, passes_size_gate, stage1_routing
from ltx_trainer.fg_vehicle_vit.stage2 import ViTClassifierHead


@dataclass
class PipelineOutput:
    fine_label: str
    stage1_label: str
    stage2_confidence: float | None
    abstained: bool
    routed_to_vit: bool


def run_two_stage_on_detection(
    det: Detection,
    frame_height: int,
    frame_width: int,
    crop_tensor: torch.Tensor | None,
    vit: ViTClassifierHead,
    cfg: FgVehicleVitConfig | None = None,
) -> PipelineOutput:
    """Apply Fig. 1 routing for one detection."""
    cfg = cfg or FgVehicleVitConfig()
    route = stage1_routing(det)
    if route == "passthrough":
        return PipelineOutput(
            fine_label=det.coco_label,
            stage1_label=det.coco_label,
            stage2_confidence=None,
            abstained=False,
            routed_to_vit=False,
        )
    if route == "discard":
        return PipelineOutput(
            fine_label=UNKNOWN_LABEL,
            stage1_label=det.coco_label,
            stage2_confidence=None,
            abstained=True,
            routed_to_vit=False,
        )
    if not passes_size_gate(det, frame_height, frame_width, cfg):
        return PipelineOutput(
            fine_label=det.coco_label,
            stage1_label=det.coco_label,
            stage2_confidence=None,
            abstained=False,
            routed_to_vit=False,
        )
    if crop_tensor is None:
        return PipelineOutput(
            fine_label=UNKNOWN_LABEL,
            stage1_label=det.coco_label,
            stage2_confidence=0.0,
            abstained=True,
            routed_to_vit=True,
        )
    logits = vit(crop_tensor.unsqueeze(0) if crop_tensor.dim() == 3 else crop_tensor)[0]
    label, conf, abst = classify_with_abstention(logits, cfg)
    return PipelineOutput(
        fine_label=label,
        stage1_label=det.coco_label,
        stage2_confidence=conf,
        abstained=abst,
        routed_to_vit=True,
    )


def pipeline_smoke(cfg: FgVehicleVitConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FgVehicleVitConfig()
    vit = ViTClassifierHead(cfg)
    dets = [
        Detection((10, 10, 200, 150), 2, 0.9),  # car → vit
        Detection((10, 10, 80, 60), 2, 0.4),  # filtered conf
        Detection((10, 10, 300, 280), 5, 0.8),  # bus passthrough
        Detection((0, 0, 5, 5), 2, 0.95),  # too small
    ]
    kept = filter_detections(dets, cfg)
    crop = torch.randn(3, cfg.crop_size, cfg.crop_size)
    outputs = [
        run_two_stage_on_detection(d, frame_height=1080, frame_width=1920, crop_tensor=crop, vit=vit, cfg=cfg)
        for d in kept
    ]
    return {
        "num_input": len(dets),
        "num_kept": len(kept),
        "outputs": [o.fine_label for o in outputs],
        "vit_routed": sum(o.routed_to_vit for o in outputs),
    }
