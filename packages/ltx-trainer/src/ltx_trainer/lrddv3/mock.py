"""LRDDv3 range + benchmark smoke (arXiv:2605.25942)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    range_mod = load_sibling(__file__, "range")
    cfg = cfg or cfg_mod.LRDDv3Config()
    drone_range_m = range_mod.drone_range_m
    haversine_horizontal_m = range_mod.haversine_horizontal_m
    r = drone_range_m(
        39.95,
        -75.19,
        120.0,
        39.951,
        -75.189,
        115.0,
        earth_radius_m=cfg.earth_radius_m,
    )
    d_h = haversine_horizontal_m(39.95, -75.19, 39.951, -75.189, earth_radius_m=cfg.earth_radius_m)

    out: dict[str, Any] = {
        "paper": "arXiv:2605.25942",
        "range_sample_m": round(r, 2),
        "haversine_horizontal_m": round(d_h, 2),
        "num_rgb": float(cfg.num_rgb_images),
        "lrddv3_map50_detfly": 0.484,
    }

    try:
        import torch
        batch_drone_range_m = range_mod.batch_drone_range_m
        pipe = load_sibling(__file__, "pipeline")

        lat_c = torch.tensor([39.95, 39.96])
        lon_c = torch.tensor([-75.19, -75.18])
        alt_c = torch.tensor([120.0, 121.0])
        lat_t = torch.tensor([39.951, 39.961])
        lon_t = torch.tensor([-75.189, -75.179])
        alt_t = torch.tensor([115.0, 116.0])
        batch_r = batch_drone_range_m(
            lat_c, lon_c, alt_c, lat_t, lon_t, alt_t, earth_radius_m=cfg.earth_radius_m
        )
        out["batch_range_mean_m"] = round(float(batch_r.mean()), 2)
        out.update({k: v for k, v in pipe.evaluation_demo().items() if k not in out})
    except ImportError:
        pass

    return out
