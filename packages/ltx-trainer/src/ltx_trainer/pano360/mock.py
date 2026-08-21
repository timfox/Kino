"""Computed pano360 evaluation smoke (geometry + optional pinhole extract)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pano360.geometry import (
    direction_to_equirect_uv,
    equirect_pixel_to_direction,
    is_equirectangular_size,
)
from ltx_trainer.pano360.synthetic import synthesize_equirect_rgb
from ltx_trainer.pano360.views import ViewSpec, default_ring_views, extract_pinhole_view


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    h, w = 128, 256
    eq = is_equirectangular_size(w, h)
    rgb = synthesize_equirect_rgb(h, w, seed=seed)

    # Direction roundtrip at center pixel
    d0 = equirect_pixel_to_direction(w * 0.5, h * 0.5, w, h)
    u, v = direction_to_equirect_uv(d0, w, h)
    roundtrip_err = float(np.linalg.norm(np.array([u, v]) - np.array([w * 0.5, h * 0.5])))

    views = default_ring_views(8, extra_pitches=(-25.0, 25.0))
    out: dict[str, Any] = {
        "package": "pano360",
        "is_equirectangular": eq,
        "erp_size": [h, w],
        "direction_roundtrip_px_err": round(roundtrip_err, 4),
        "view_count": len(views),
    }

    try:
        spec = ViewSpec(yaw_deg=45.0, pitch_deg=0.0, fov_deg=90.0)
        pinhole = extract_pinhole_view(rgb, spec, out_width=64, out_height=36)
        out.update(
            {
                "opencv": True,
                "pinhole_shape": list(pinhole.shape),
                "pinhole_mean": round(float(pinhole.mean()), 2),
            }
        )
    except ImportError:
        out["opencv"] = False

    return out
