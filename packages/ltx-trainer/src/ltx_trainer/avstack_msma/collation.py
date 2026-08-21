"""Dataset interfaces and COCO export path (§2.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.avstack_msma.scenario import ScenarioSpec


@dataclass
class CocoImageRecord:
    file_name: str
    width: int
    height: int
    agent: str
    sensor: str
    frame_id: int


@dataclass
class CocoAnnotation:
    image_id: int
    category: str
    bbox_xywh: tuple[float, float, float, float]
    object_id: int


@dataclass
class CocoExport:
    images: list[CocoImageRecord] = field(default_factory=list)
    annotations: list[CocoAnnotation] = field(default_factory=list)
    categories: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [img.__dict__ for img in self.images],
            "annotations": [
                {
                    "image_id": a.image_id,
                    "category": a.category,
                    "bbox": list(a.bbox_xywh),
                    "object_id": a.object_id,
                }
                for a in self.annotations
            ],
            "categories": self.categories,
        }


def build_coco_skeleton(spec: ScenarioSpec, *, max_frames_per_sensor: int = 3) -> CocoExport:
    """Iterate scenes/agents/sensors/frames — COCO folder layout stub."""
    export = CocoExport(
        categories=[
            {"id": 1, "name": "car"},
            {"id": 2, "name": "truck"},
            {"id": 3, "name": "bicycle"},
            {"id": 4, "name": "motorcycle"},
        ]
    )
    image_id = 0
    for actor in spec.actors:
        for s_idx, sensor in enumerate(actor.sensors):
            if sensor.modality.value not in ("rgb", "depth", "semantic"):
                continue
            for frame in range(max_frames_per_sensor):
                export.images.append(
                    CocoImageRecord(
                        file_name=f"{actor.name}/{sensor.modality.value}/frame_{frame:06d}.png",
                        width=1920,
                        height=1080,
                        agent=actor.name,
                        sensor=sensor.modality.value,
                        frame_id=frame,
                    )
                )
                export.annotations.append(
                    CocoAnnotation(
                        image_id=image_id,
                        category="car",
                        bbox_xywh=(400.0, 300.0, 120.0, 80.0),
                        object_id=frame + s_idx,
                    )
                )
                image_id += 1
    return export


def downstream_training_targets() -> list[dict[str, str]]:
    """MMDetection / MMDetection3D / AVstack plugin paths (§4.1)."""
    return [
        {"library": "MMDetection", "tasks": "2D detection"},
        {"library": "MMDetection3D", "tasks": "LiDAR 3D detection"},
        {"library": "AVstack", "tasks": "native dataset managers + model plugins"},
        {"library": "Frustum PointNets / SECOND / PointPillars", "tasks": "representative 3D pipelines"},
    ]
