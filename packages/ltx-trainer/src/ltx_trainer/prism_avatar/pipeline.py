"""End-to-end PrismAvatar CPU pipeline."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ltx_trainer.prism_avatar.config import LIVE_FPS, PANEL_RES, STUDENT_FPS, VIEW_RES, DISPLAY_VIEWS, PrismAvatarConfig
from ltx_trainer.prism_avatar.driver import distill_controls_from_frame, runtime_profile_ms
from ltx_trainer.prism_avatar.losses import pmv_gamma, pmv_loss_bundle
from ltx_trainer.prism_avatar.masks import head_hair_matte, semantic_support_masks
from ltx_trainer.prism_avatar.metrics import side_artifact_metrics
from ltx_trainer.prism_avatar.pmv import alignment_gate, select_pmv_frames, virtual_camera
from ltx_trainer.prism_avatar.subpixel import display_view_yaws, encode_panel_raster
from ltx_trainer.prism_avatar.synthetic import synthetic_turn_sequence


def run_training_smoke(cfg: PrismAvatarConfig | None = None, *, seed: int = 0) -> dict:
    cfg = cfg or PrismAvatarConfig()
    data = synthetic_turn_sequence(cfg, seed=seed)
    pmv_frames = select_pmv_frames(data["yaws"], data["scores"], cfg)
    support = semantic_support_masks(data["positive"], data["risk"], cfg)
    bottom = np.ones_like(data["alpha"])
    matte = head_hair_matte(data["alpha"], support, bottom, cfg)
    align_ok = alignment_gate(0.45, 0.18, 12.0, cfg)
    losses = pmv_loss_bundle(
        data["rgb"],
        data["alpha"],
        data["rgb"] * 0.98,
        data["alpha"] * 0.99,
        data["mesh"],
        matte,
        cfg,
        stage="side",
    )
    gamma = pmv_gamma(cfg.pmv_start_iter + 350, cfg)
    return {
        "pmv_frames": len(pmv_frames),
        "align_ok": align_ok,
        "gamma": gamma,
        "losses": losses,
        "virtual_cam_25": virtual_camera(25.0, cfg),
    }


def smoke_config() -> PrismAvatarConfig:
    """Reduced resolution for CPU smoke (paper defaults remain on PrismAvatarConfig)."""
    return PrismAvatarConfig(
        view_height=64,
        view_width=64,
        panel_height=128,
        panel_width=128,
        num_views=8,
    )


def run_display_smoke(cfg: PrismAvatarConfig | None = None, *, seed: int = 0) -> dict:
    cfg = cfg or smoke_config()
    rng = np.random.default_rng(seed)
    vh, vw = min(64, cfg.view_height), min(64, cfg.view_width)
    views = [rng.random((vh, vw, 3)) for _ in range(cfg.num_views)]
    panel = encode_panel_raster(views, cfg)
    yaws = display_view_yaws(cfg)
    teacher = rng.random(129)
    frame = rng.random((32, 32, 3))
    pred, dloss = distill_controls_from_frame(frame, teacher, seed=seed)
    live = runtime_profile_ms(student=False)
    student = runtime_profile_ms(student=True)
    return {
        "panel_shape": list(panel.shape),
        "view_yaws": [float(yaws[0]), float(yaws[-1])],
        "distill_loss": dloss,
        "control_dim": int(pred.size),
        "live_fps": live["fps"],
        "student_fps": student["fps"],
    }


def run_marcel_metrics_smoke(cfg: PrismAvatarConfig | None = None, *, seed: int = 0) -> dict:
    cfg = cfg or PrismAvatarConfig()
    data = synthetic_turn_sequence(cfg, seed=seed)
    metrics = side_artifact_metrics(data["rgb"], data["alpha"], data["mesh"])
    return {"marcel_probe": metrics}


def evaluation_smoke(*, seed: int = 0) -> dict:
    """Paper stub entrypoint."""
    cfg = smoke_config()
    train = run_training_smoke(cfg, seed=seed)
    display = run_display_smoke(cfg, seed=seed)
    marcel = run_marcel_metrics_smoke(cfg, seed=seed)
    return {
        "paper": "PrismAvatar",
        "arxiv": "2606.10550",
        "train": train,
        "display": display,
        "marcel": marcel,
        "fps_anchors": {"live": LIVE_FPS, "student": STUDENT_FPS},
        "display_config_full": {
            "views": DISPLAY_VIEWS,
            "view_res": list(VIEW_RES),
            "panel_res": list(PANEL_RES),
        },
        "config": asdict(cfg),
    }
