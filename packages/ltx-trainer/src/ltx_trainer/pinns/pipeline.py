"""PINNS framework card, dataset comparison, and baseline tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.pinns.config import PINNSConfig
from ltx_trainer.pinns.homography import (
    apply_homography,
    estimate_homography,
    reconstruction_error_m,
    reprojection_error,
)
from ltx_trainer.pinns.metrics import ade, batch_ade_fde, fde


def framework_card(cfg: PINNSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PINNSConfig()
    return {
        "name": "PINNS",
        "paper": "arXiv:2605.25947",
        "title": "Pedestrian–Vehicle Interaction Benchmark via Uncalibrated Cameras",
        "full_name": "Pedestrian–vehicle Interaction dataset from uNcalibrated cameras in uNstructured Scenes",
        "acquisition": "uncalibrated fixed surveillance cameras",
        "calibration": "satellite-map homography to BEV",
        "scenes": list(cfg.scenes),
        "objects": {
            "pedestrians": cfg.num_pedestrians,
            "vehicles": cfg.num_vehicles,
            "trajectory_points": cfg.trajectory_points_ped + cfg.trajectory_points_veh,
        },
        "benchmark": f"{cfg.baseline_model}, {cfg.t_obs} obs / {cfg.t_pred} pred @ {cfg.eval_hz} Hz",
        "repo": cfg.repo,
    }


def table_dataset_comparison() -> dict[str, dict[str, str | bool]]:
    """Table I — representative datasets vs PINNS (abbreviated flags)."""
    return {
        "eth": {"unstructured": True, "agents": "ped", "interaction": "ped-ped"},
        "ucy": {"unstructured": True, "agents": "ped", "interaction": "ped-ped"},
        "sdd": {"unstructured": True, "agents": "ped", "interaction": "ped-ped"},
        "hbs": {"unstructured": True, "agents": "ped+veh", "interaction": "ped-veh"},
        "citr": {"unstructured": True, "agents": "ped+veh", "interaction": "ped-veh"},
        "nuscenes": {"unstructured": False, "agents": "veh", "interaction": "-"},
        "interaction_ds": {"unstructured": False, "agents": "veh", "interaction": "veh-veh"},
        "pinns": {
            "unstructured": True,
            "agents": "ped+veh",
            "interaction": "ped-veh",
            "acquisition": "uncalibrated_cameras",
            "weather": "multi",
        },
    }


def table_baseline_trajectron() -> dict[str, dict[str, float]]:
    """Table II — Trajectron++ ADE/FDE on PINNS vs prior datasets."""
    return {
        "eth": {"ade": 0.71, "fde": 1.66},
        "hotel": {"ade": 0.22, "fde": 0.46},
        "univ": {"ade": 0.44, "fde": 1.17},
        "zara1": {"ade": 0.30, "fde": 0.79},
        "zara2": {"ade": 0.23, "fde": 0.59},
        "nuscenes_ped": {"ade": float("nan"), "fde": 0.60},
        "nuscenes_veh": {"ade": float("nan"), "fde": 2.20},
        "pinns_ped": {"ade": 0.96, "fde": 1.93},
        "pinns_veh": {"ade": 1.90, "fde": 3.79},
    }


def table_calibration_stats() -> dict[str, float]:
    """Sec. III-B reported homography accuracy."""
    return {
        "reprojection_error_px": 8.24,
        "reconstruction_error_m": 0.28,
        "reference_points_min": 10.0,
        "reference_points_max": 15.0,
    }


def training_step_demo(
    cfg: PINNSConfig | None = None,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Smoke: homography fit + synthetic trajectory ADE/FDE."""
    cfg = cfg or PINNSConfig()
    dev = torch.device(device)

    img = torch.tensor(
        [[100.0, 200.0], [500.0, 220.0], [900.0, 400.0], [300.0, 700.0], [800.0, 750.0]],
        device=dev,
    )
    world = torch.tensor(
        [[0.0, 0.0], [10.0, 0.2], [20.0, 5.0], [5.0, 15.0], [18.0, 16.0]],
        device=dev,
    )
    h = estimate_homography(img, world)
    reproj = reprojection_error(img, world, h)
    world_hat = apply_homography(img, h)
    recon = reconstruction_error_m(world_hat, world)

    t = cfg.t_pred
    gt = torch.cumsum(torch.randn(t, 2, device=dev) * 0.3, dim=0)
    pred = gt + torch.randn(t, 2, device=dev) * 0.5
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)

    ade_b, fde_b = batch_ade_fde(pred_batch, gt_batch)
    return {
        "reprojection_error_px": reproj,
        "reconstruction_error_m": recon,
        "ade_sample": ade_b,
        "fde_sample": fde_b,
        "homography_det": float(torch.linalg.det(h[:2, :2])),
        "num_scenes": float(cfg.num_scenes),
    }


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    step = training_step_demo(device=device)
    base = table_baseline_trajectron()
    cal = table_calibration_stats()
    return {
        **step,
        "pinns_ped_ade": base["pinns_ped"]["ade"],
        "pinns_ped_fde": base["pinns_ped"]["fde"],
        "pinns_veh_ade": base["pinns_veh"]["ade"],
        "pinns_veh_fde": base["pinns_veh"]["fde"],
        "eth_ade": base["eth"]["ade"],
        "paper_reproj_px": cal["reprojection_error_px"],
        "veh_fde_vs_ped_ratio": base["pinns_veh"]["fde"] / base["pinns_ped"]["fde"],
    }
