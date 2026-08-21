"""Benchmark datasets (Sec. 4.1)."""

from __future__ import annotations

from typing import Any


def datasets_card() -> dict[str, Any]:
    return {
        "depth_camera": {
            "finetune": "ScanNet train — 3K RGB–depth pairs (coupled IPQ, DPQ)",
            "test": ["ScanNet (50-frame pairs)", "3DMatch (40-frame pairs)"],
            "metrics": ["rotation acc/error", "translation acc/error", "Chamfer"],
        },
        "lidar": {
            "finetune": "Dur360BEV — ~10K coupled panorama + range-map pairs",
            "test": "Dur360BEV — timestamps 11510–13149; pairs ≥5m / ≥10m",
            "representation": "equirectangular range map → panoramic RGB",
            "metrics": ["FMR", "IR", "RR"],
        },
        "lidar_dataset_fov_table1": {
            "KITTI": {"azimuth": "<360", "polar": "<180"},
            "KITTI-360": {"azimuth": "360", "polar": "120"},
            "Waymo": {"azimuth": "<360", "polar": "<180"},
            "nuScenes": {"azimuth": "360", "polar": "40"},
            "Dur360BEV": {"azimuth": "360", "polar": "180"},
        },
        "baselines": ["FCGF", "Predator", "GeoTrans", "FPFH", "ColorPCR", "CoFiNet", "PARE-Net"],
    }
