"""Runnable evaluation smoke for expo_cm."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    try:
        import torch  # noqa: F401

        return _torch_smoke()
    except ImportError:
        return _numpy_smoke()


def _torch_smoke() -> dict[str, Any]:
    traj = load_sibling(__file__, "trajectory")
    import torch
    ldr = torch.rand(1, 3, 32, 32)
    y0 = ldr * 0.85
    xt, masks = traj.sample_eact_state(ldr, y0, t=0.5)
    return {
        "package": "expo_cm",
        "xt_mean": round(float(xt.mean()), 4),
        "wgood_frac": round(float(masks["wgood"].mean()), 4),
    }


def _numpy_smoke() -> dict[str, Any]:
    import numpy as np
    ldr = np.random.default_rng(0).random((32, 32))
    low = ldr * 0.5
    high = np.clip(ldr * 1.8, 0, 1)
    return {"package": "expo_cm", "under_mean": round(float(low.mean()), 4), "over_mean": round(float(high.mean()), 4)}
