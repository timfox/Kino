"""LRDDv3 framework card, dataset comparison, and benchmark tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lrddv3.config import LRDDv3Config
from ltx_trainer.lrddv3.range import batch_drone_range_m, drone_range_m


def framework_card(cfg: LRDDv3Config | None = None) -> dict[str, Any]:
    cfg = cfg or LRDDv3Config()
    return {
        "name": "LRDDv3",
        "paper": "arXiv:2605.25942",
        "title": "High-Resolution Long-Range Drone Detection Dataset with Range and Thermal Data",
        "rgb_images": cfg.num_rgb_images,
        "ir_images": cfg.num_ir_images,
        "rgb_resolution": f"{cfg.rgb_resolution[0]}x{cfg.rgb_resolution[1]}",
        "ir_resolution": f"{cfg.ir_resolution[0]}x{cfg.ir_resolution[1]}",
        "sampling": f"{cfg.sample_fps} FPS from {cfg.num_video_clips} clips",
        "range_m": [cfg.range_min_m, cfg.range_max_m],
        "annotations": {
            "drone": cfg.annotations_drone,
            "bird": cfg.annotations_bird,
            "airplane": cfg.annotations_airplane,
        },
        "collection": f"{cfg.collection_days} days over {cfg.collection_months} months",
        "url": cfg.dataset_url,
    }


def table_dataset_comparison() -> dict[str, dict[str, str | int | bool]]:
    """Table I — LRDDv3 vs representative drone detection datasets."""
    return {
        "multi_sensor": {
            "images": 203_328,
            "modalities": "RGB,IR,Audio",
            "resolution": "640x512",
            "range": "categorical",
            "mobile_camera": False,
        },
        "bir_drone": {"images": 3_300, "modalities": "RGB", "resolution": "640x640", "range": "no"},
        "visiodect": {"images": 20_924, "modalities": "RGB,EO-IR", "resolution": "852x480", "range": "no"},
        "det_fly": {"images": 13_300, "modalities": "RGB", "resolution": "4K", "mobile_camera": True},
        "lrddv1": {
            "images": 21_200,
            "modalities": "RGB",
            "resolution": "1920x1080",
            "weather": True,
            "range": "no",
        },
        "lrddv2": {
            "images": 39_516,
            "modalities": "RGB",
            "resolution": "1920x1080",
            "range": "partial",
        },
        "lrddv3": {
            "images": 102_532,
            "ir_images": 29_630,
            "modalities": "RGB,IR",
            "resolution": "4K / 640x512",
            "range": "0-200m",
            "mobile_camera": True,
            "weather": True,
            "lighting": True,
            "occlusion": True,
        },
    }


def table_detfly_yolov11_benchmark() -> dict[str, dict[str, float]]:
    """Table II — YOLOv11m trained on dataset, tested on Det-Fly."""
    return {
        "drone_vs_bird": {"map50": 0.484, "map50_95": 0.214, "f1": 0.492},
        "dut_anti_uav": {"map50": 0.431, "map50_95": 0.213, "f1": 0.485},
        "lrddv1_v2": {"map50": 0.330, "map50_95": 0.121, "f1": 0.356},
        "lrddv3": {"map50": 0.485, "map50_95": 0.261, "f1": 0.468},
    }


def table_resolution_benchmark() -> dict[str, dict[str, float]]:
    """Table III — YOLOv11m on LRDDv3 test set at two input sizes."""
    return {
        "640x640": {"map50": 0.543, "map50_95": 0.288, "f1": 0.482},
        "1920x1920": {"map50": 0.822, "map50_95": 0.504, "f1": 0.800},
    }


def split_summary() -> dict[str, int]:
    """Fig. 3(a) — approximate official split sizes (paper cites 20k test RGB)."""
    return {
        "test_rgb": 20_000,
        "test_clips": 34,
        "train_val_rgb": 102_532 - 20_000,
    }


def training_step_demo(cfg: LRDDv3Config | None = None) -> dict[str, float]:
    """Smoke: Haversine range + synthetic batch ranges."""
    cfg = cfg or LRDDv3Config()
    r = drone_range_m(39.95, -75.19, 120.0, 39.951, -75.189, 115.0, earth_radius_m=cfg.earth_radius_m)

    lat_c = torch.tensor([39.95, 40.0])
    lon_c = torch.tensor([-75.19, -75.2])
    alt_c = torch.tensor([100.0, 80.0])
    lat_t = torch.tensor([39.951, 40.01])
    lon_t = torch.tensor([-75.189, -75.19])
    alt_t = torch.tensor([95.0, 85.0])
    batch = batch_drone_range_m(lat_c, lon_c, alt_c, lat_t, lon_t, alt_t, earth_radius_m=cfg.earth_radius_m)

    return {
        "range_sample_m": r,
        "batch_mean_range_m": float(batch.mean()),
        "num_rgb": float(cfg.num_rgb_images),
        "num_ir": float(cfg.num_ir_images),
        "min_bbox_px": float(cfg.min_bbox_pixels),
    }


def evaluation_demo() -> dict[str, Any]:
    detfly = table_detfly_yolov11_benchmark()
    res = table_resolution_benchmark()
    lrdd = detfly["lrddv3"]
    base = detfly["lrddv1_v2"]
    hi = res["1920x1920"]
    lo = res["640x640"]
    return {
        **training_step_demo(),
        "lrddv3_map50_detfly": lrdd["map50"],
        "lrddv3_map50_95_detfly": lrdd["map50_95"],
        "map50_gain_vs_lrddv1v2": lrdd["map50"] - base["map50"],
        "map50_95_best_on_detfly": lrdd["map50_95"] >= detfly["drone_vs_bird"]["map50_95"],
        "hi_res_map50": hi["map50"],
        "hi_res_gain_map50": hi["map50"] - lo["map50"],
        "annotation_total": float(
            LRDDv3Config().annotations_drone
            + LRDDv3Config().annotations_bird
            + LRDDv3Config().annotations_airplane
        ),
    }
