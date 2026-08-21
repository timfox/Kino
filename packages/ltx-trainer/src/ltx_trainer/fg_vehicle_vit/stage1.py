"""Stage 1 RT-DETR detection gates (§2.4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig
from ltx_trainer.fg_vehicle_vit.taxonomy import COCO_ID_TO_NAME, route_stage2


@dataclass
class Detection:
    """Single axis-aligned vehicle detection."""

    bbox_xyxy: tuple[float, float, float, float]
    coco_class_id: int
    confidence: float

    @property
    def coco_label(self) -> str:
        return COCO_ID_TO_NAME.get(self.coco_class_id, "unknown")


def filter_detections(
    detections: list[Detection],
    cfg: FgVehicleVitConfig | None = None,
) -> list[Detection]:
    """Retain COCO vehicle classes above Stage 1 confidence threshold."""
    cfg = cfg or FgVehicleVitConfig()
    allowed = set(cfg.coco_vehicle_class_ids)
    return [
        d
        for d in detections
        if d.coco_class_id in allowed and d.confidence >= cfg.stage1_conf_threshold
    ]


def passes_size_gate(
    det: Detection,
    frame_height: int,
    frame_width: int,
    cfg: FgVehicleVitConfig | None = None,
) -> bool:
    """Minimum box size before Stage 2 (§2.4)."""
    cfg = cfg or FgVehicleVitConfig()
    x1, y1, x2, y2 = det.bbox_xyxy
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    short_side = min(w, h)
    area = w * h
    frame_area = frame_height * frame_width
    if short_side < cfg.min_box_short_side_frac * frame_height:
        return False
    if area < cfg.min_box_area_frac * frame_area:
        return False
    return True


def stage1_routing(det: Detection) -> str:
    return route_stage2(det.coco_label)
