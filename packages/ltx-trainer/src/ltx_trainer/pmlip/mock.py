"""Runnable evaluation smoke for pmlip."""

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
    crps_mod = load_sibling(__file__, "crps")
    import torch
    samples = torch.linspace(0.0, 1.0, 8)
    loss = float(crps_mod.fair_crps(samples, torch.tensor(0.55)).item())
    return {"package": "pmlip", "fair_crps": round(loss, 4)}


def _numpy_smoke() -> dict[str, Any]:
    import numpy as np
    samples = np.linspace(0, 1, 8)
    target = 0.55
    rel = np.abs(samples - target).mean()
    spread = np.abs(samples[:, None] - samples[None, :]).mean()
    score = rel - 0.5 * spread
    return {"package": "pmlip", "fair_crps": round(float(score), 4)}
