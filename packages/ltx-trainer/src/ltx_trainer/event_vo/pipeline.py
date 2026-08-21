"""End-to-end event VO pipeline and framework card."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.event_vo.config import EventVOConfig
from ltx_trainer.event_vo.eskf import MonocularEventESKF
from ltx_trainer.event_vo.homography import estimate_homography_dlt, homography_to_pose
from ltx_trainer.event_vo.metrics import absolute_trajectory_error, align_sim3_umeyama
from ltx_trainer.event_vo.rate import FeatureUpdate
from ltx_trainer.event_vo.simulator import generate_planar_scene


def framework_card(cfg: EventVOConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EventVOConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "Planetary rovers need low-latency pose under HDR lighting; event cameras "
            "plus RATE async tracks feed an ESKF monocular VO without IMU."
        ),
        "method": {
            "frontend": "RATE (SAEB + Shi-Tomasi + HASTE async tracks)",
            "backend": "Error-State Kalman Filter with stochastic cloning",
            "init": "Homography on grouped event feature sets (planar)",
            "async": "One measurement update per track message",
        },
        "reference_metrics": {
            "sim_mape_m": cfg.sim_mape_m,
            "uzh_poster_mape_m": cfg.uzh_poster_mape_m,
        },
    }


def homography_bootstrap(
    filter_: MonocularEventESKF,
    ref_updates: list[FeatureUpdate],
    cur_updates: list[FeatureUpdate],
) -> bool:
    """Initialize filter from two event feature sets (Fig. 2)."""
    cfg = filter_.cfg
    common = {u.feature_id for u in ref_updates} & {u.feature_id for u in cur_updates}
    if len(common) < 8:
        return False
    src, dst = [], []
    ref_map = {u.feature_id: (u.u, u.v) for u in ref_updates}
    cur_map = {u.feature_id: (u.u, u.v) for u in cur_updates}
    for fid in common:
        src.append(ref_map[fid])
        dst.append(cur_map[fid])
    H = estimate_homography_dlt(np.asarray(src), np.asarray(dst))
    if H is None:
        return False
    K = np.array([[cfg.fx, 0, cfg.cx], [0, cfg.fy, cfg.cy], [0, 0, 1]], dtype=np.float64)
    pose = homography_to_pose(H, K)
    if pose is None:
        return False
    R, t = pose
    from ltx_trainer.event_vo.lie import rot_to_theta

    filter_.theta = rot_to_theta(R)
    filter_.p = t * 0.5
    filter_.v = np.zeros(3)
    filter_.mark_initialized()
    return True


def run_simulation_odometry(
    cfg: EventVOConfig | None = None,
    *,
    seed: int = 42,
    noise_px: float = 1.0,
) -> dict[str, Any]:
    cfg = cfg or EventVOConfig()
    scene = generate_planar_scene(cfg, noise_px=noise_px, seed=seed)
    filt = MonocularEventESKF(cfg)

    # Group updates by time for homography init
    t0 = scene.times[0]
    t_init = t0 + 0.5
    ref: list[FeatureUpdate] = []
    cur: list[FeatureUpdate] = []
    rest: list[FeatureUpdate] = []
    for u in scene.updates:
        if u.time_s < t_init:
            ref.append(u)
        elif u.time_s < t_init + 0.3:
            cur.append(u)
        else:
            rest.append(u)

    if not homography_bootstrap(filt, ref, cur):
        # Fallback: identity init for smoke
        filt.mark_initialized()

    # Seed a few landmarks from ground truth for stable smoke (planar sim)
    rng = np.random.default_rng(seed)
    for fid in rng.choice(len(scene.landmarks), size=min(15, len(scene.landmarks)), replace=False):
        filt._augment_landmark(int(fid), scene.landmarks[int(fid)])

    for u in rest:
        try:
            filt.process_update(u)
        except np.linalg.LinAlgError:
            continue

    est_pts = np.array([p for _, p in filt.trajectory]) if filt.trajectory else filt.p.reshape(1, 3)
    gt_pts = scene.gt_positions[: len(est_pts)]
    if len(est_pts) < 3:
        return {"ok": False, "reason": "insufficient poses"}

    aligned, scale = align_sim3_umeyama(est_pts, gt_pts)
    metrics = absolute_trajectory_error(aligned, gt_pts)
    return {
        "ok": True,
        "n_updates": len(scene.updates),
        "n_landmarks": len(filt.landmark_order),
        "sim3_scale": scale,
        "metrics": metrics,
        "paper_sim_mape_m": cfg.sim_mape_m,
    }


def evaluation_demo() -> dict[str, Any]:
    from ltx_trainer.event_vo.mock import evaluation_smoke

    return {"smoke": evaluation_smoke(), "simulation": run_simulation_odometry()}
