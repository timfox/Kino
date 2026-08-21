"""Sphere-Depth evaluation pipeline."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.sphere_depth.benchmarks import DEPTH_ANYTHING_CUBEMAP, table2_by_model
from ltx_trainer.sphere_depth.calibration import landmark_mse, learn_scaling_lambda, split_landmarks
from ltx_trainer.sphere_depth.config import SphereDepthConfig
from ltx_trainer.sphere_depth.cubemap import cubemap_faces_to_erp, erp_to_cubemap_faces
from ltx_trainer.sphere_depth.models import build_estimator, predict_landmark_depths
from ltx_trainer.sphere_depth.pose import apply_pose_perturbation
from ltx_trainer.sphere_depth.synthetic import synthetic_scene


def evaluate_model_on_scene(
    model_name: str,
    scene: dict[str, torch.Tensor],
    cfg: SphereDepthConfig,
) -> dict[str, float]:
    """Calibrate λ on train landmarks; report ε on test landmarks."""
    estimator = build_estimator(model_name)
    erp = scene["erp"]
    u, v, gt = scene["landmark_u"], scene["landmark_v"], scene["gt_depth"]
    depth_map = estimator(erp)
    pred_all = predict_landmark_depths(depth_map, u.unsqueeze(0), v.unsqueeze(0))
    train_idx, test_idx = split_landmarks(gt.numel(), cfg.train_landmark_ratio)
    lam = learn_scaling_lambda(pred_all[train_idx], gt[train_idx])
    eps_test = landmark_mse(pred_all[test_idx], gt[test_idx], lam)
    return {"lambda": lam, "epsilon_test": eps_test}


def evaluate_pose_sensitivity(
    model_name: str,
    scene: dict[str, torch.Tensor],
    cfg: SphereDepthConfig,
    pitch_deg: float,
    roll_deg: float,
) -> float:
    estimator = build_estimator(model_name)
    erp = apply_pose_perturbation(scene["erp"], pitch_deg, roll_deg)
    u, v, gt = scene["landmark_u"], scene["landmark_v"], scene["gt_depth"]
    depth_map = estimator(erp)
    pred = predict_landmark_depths(depth_map, u.unsqueeze(0), v.unsqueeze(0))
    train_idx, test_idx = split_landmarks(gt.numel(), cfg.train_landmark_ratio)
    lam = learn_scaling_lambda(pred[train_idx], gt[train_idx])
    return landmark_mse(pred[test_idx], gt[test_idx], lam)


def evaluate_depth_anything_cubemap(
    scene: dict[str, torch.Tensor],
    cfg: SphereDepthConfig,
) -> dict[str, float]:
    """Sec. 3.2 — cubemap faces + Depth Anything stub."""
    from ltx_trainer.sphere_depth.disparity import relative_depth_stub

    faces = erp_to_cubemap_faces(scene["erp"], cfg.cubemap_face_size)
    face_depths = {k: relative_depth_stub(v) for k, v in faces.items()}
    erp_depth = cubemap_faces_to_erp(face_depths, cfg.erp_height, cfg.erp_width)
    u, v, gt = scene["landmark_u"], scene["landmark_v"], scene["gt_depth"]
    pred = predict_landmark_depths(erp_depth.unsqueeze(0), u.unsqueeze(0), v.unsqueeze(0))
    train_idx, test_idx = split_landmarks(gt.numel(), cfg.train_landmark_ratio)
    lam = learn_scaling_lambda(pred[train_idx], gt[train_idx])
    eps = landmark_mse(pred[test_idx], gt[test_idx], lam)
    return {"lambda": lam, "epsilon_test": eps}


def evaluation_demo_run(cfg: SphereDepthConfig | None = None, *, device: str = "cpu") -> dict[str, Any]:
    cfg = cfg or SphereDepthConfig(erp_height=64, erp_width=128, cubemap_face_size=32)
    dev = torch.device(device)
    scene = synthetic_scene(cfg, num_landmarks=24, device=dev)
    results: dict[str, Any] = {}
    for name in ("ACDNet", "DepthAnywhere", "SliceNet"):
        results[name] = evaluate_model_on_scene(name, scene, cfg)
    results["pose_acdnet_15deg"] = evaluate_pose_sensitivity("ACDNet", scene, cfg, 15.0, 15.0)
    results["depth_anything_cubemap"] = evaluate_depth_anything_cubemap(scene, cfg)
    ref = table2_by_model()
    results["ref_acdnet_gravity"] = ref["ACDNet"]["gravity_aligned"]
    results["ref_cubemap"] = DEPTH_ANYTHING_CUBEMAP["gravity_aligned"]
    return results
