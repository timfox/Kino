"""Six-class vocabulary, COCO routing, and label normalization (§2.1, §2.7)."""

from __future__ import annotations

from ltx_trainer.fg_vehicle_vit.config import FgVehicleVitConfig

# COCO coarse names for Stage 1 vehicle IDs (2=car, 3=motorcycle, 5=bus, 7=truck)
COCO_ID_TO_NAME: dict[int, str] = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Stage 2 applies only to car/truck detections per Fig. 1
STAGE2_ELIGIBLE_COCO: frozenset[str] = frozenset({"car", "truck"})

# Ground-truth alias normalization (§2.7.1)
LABEL_ALIASES: dict[str, str] = {
    "car": "passenger car",
    "passenger_car": "passenger car",
    "suv": "SUV",
    "pickup": "pickup truck",
    "pickup_truck": "pickup truck",
    "minivan": "minivan",
    "large_van": "large van",
    "cargo van": "large van",
    "commercial_truck": "commercial truck",
    "truck": "commercial truck",
}


def normalize_label(raw: str) -> str | None:
    key = raw.strip().lower().replace("_", " ")
    if key in {c.lower() for c in FgVehicleVitConfig().fine_grained_classes}:
        for c in FgVehicleVitConfig().fine_grained_classes:
            if c.lower() == key:
                return c
    return LABEL_ALIASES.get(key)


def route_stage2(coco_label: str) -> str:
    """Return 'vit' | 'passthrough' | 'discard' for pipeline routing."""
    name = coco_label.lower()
    if name in STAGE2_ELIGIBLE_COCO:
        return "vit"
    if name in ("bus", "motorcycle"):
        return "passthrough"
    return "discard"
