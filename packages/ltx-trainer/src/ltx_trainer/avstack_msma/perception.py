"""Application-specific perception / domain shift (§4.1, Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainDomain(str, Enum):
    VEHICLE = "vehicle"
    INFRASTRUCTURE = "infrastructure"


class TestDomain(str, Enum):
    VEHICLE = "vehicle"
    INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class PerceptionAP:
    modality: str
    class_name: str
    train: TrainDomain
    test: TestDomain
    ap: float


def table_infrastructure_perception() -> list[dict[str, float | str]]:
    """Table 2 — camera viewpoint shift vs LiDAR robustness."""
    rows = [
        PerceptionAP("Camera", "Truck", TrainDomain.VEHICLE, TestDomain.VEHICLE, 0.33),
        PerceptionAP("Camera", "Truck", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("Camera", "Truck", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 0.00),
        PerceptionAP("Camera", "Motorcycle", TrainDomain.VEHICLE, TestDomain.VEHICLE, 0.00),
        PerceptionAP("Camera", "Motorcycle", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("Camera", "Motorcycle", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 0.00),
        PerceptionAP("Camera", "Car", TrainDomain.VEHICLE, TestDomain.VEHICLE, 0.51),
        PerceptionAP("Camera", "Car", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 0.67),
        PerceptionAP("Camera", "Car", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 0.00),
        PerceptionAP("Camera", "Bicycle", TrainDomain.VEHICLE, TestDomain.VEHICLE, 0.25),
        PerceptionAP("Camera", "Bicycle", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("Camera", "Bicycle", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 0.00),
        PerceptionAP("LiDAR", "Truck", TrainDomain.VEHICLE, TestDomain.VEHICLE, 1.00),
        PerceptionAP("LiDAR", "Truck", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Truck", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Motorcycle", TrainDomain.VEHICLE, TestDomain.VEHICLE, 1.00),
        PerceptionAP("LiDAR", "Motorcycle", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Motorcycle", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Car", TrainDomain.VEHICLE, TestDomain.VEHICLE, 0.65),
        PerceptionAP("LiDAR", "Car", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Car", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Bicycle", TrainDomain.VEHICLE, TestDomain.VEHICLE, 1.00),
        PerceptionAP("LiDAR", "Bicycle", TrainDomain.INFRASTRUCTURE, TestDomain.INFRASTRUCTURE, 1.00),
        PerceptionAP("LiDAR", "Bicycle", TrainDomain.VEHICLE, TestDomain.INFRASTRUCTURE, 1.00),
    ]
    return [
        {
            "modality": r.modality,
            "class": r.class_name,
            "train": r.train.value,
            "test": r.test.value,
            "ap": r.ap,
        }
        for r in rows
    ]


def domain_shift_summary() -> dict[str, float]:
    tab = table_infrastructure_perception()
    cam_cross = [r for r in tab if r["modality"] == "Camera" and r["train"] != r["test"]]
    cam_match = [r for r in tab if r["modality"] == "Camera" and r["train"] == r["test"]]
    lidar_cross = [r for r in tab if r["modality"] == "LiDAR" and r["train"] != r["test"]]
    return {
        "camera_cross_domain_mean_ap": sum(r["ap"] for r in cam_cross) / max(1, len(cam_cross)),
        "camera_matched_mean_ap": sum(r["ap"] for r in cam_match) / max(1, len(cam_match)),
        "lidar_cross_domain_mean_ap": sum(r["ap"] for r in lidar_cross) / max(1, len(lidar_cross)),
    }
