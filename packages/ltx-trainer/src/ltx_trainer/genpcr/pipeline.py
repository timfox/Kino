"""Demo / train-step stubs."""

from __future__ import annotations

from typing import Any

from ltx_trainer.genpcr.config import GenPcrConfig
from ltx_trainer.genpcr.registration_stub import GenerativePcrStub
from ltx_trainer.genpcr.synthetic import synthetic_pair


def registration_demo(cfg: GenPcrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GenPcrConfig()
    p, q = synthetic_pair(cfg)
    out = GenerativePcrStub(cfg)(p, q)
    return {
        "mode": cfg.mode,
        "match_cost": float(out["match_cost"].item()),
        "desc_dim": list(out["desc_p"].shape),
        "image_shape": list(out["image_p"].shape),
    }


def evaluation_demo_run(cfg: GenPcrConfig | None = None) -> dict[str, Any]:
    cfg_depth = cfg or GenPcrConfig(mode="depth")
    cfg_lidar = GenPcrConfig(mode="lidar", image_height=32, image_width=64)
    return {
        "depth": registration_demo(cfg_depth),
        "lidar": registration_demo(cfg_lidar),
    }
