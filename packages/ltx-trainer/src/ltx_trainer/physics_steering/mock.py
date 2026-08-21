"""Runnable evaluation smoke for physics_steering (arXiv stub)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def _round_values(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, dict):
            out[k] = _round_values(v)
        elif isinstance(v, list):
            out[k] = [round(x, 4) if isinstance(x, float) else x for x in v]
        else:
            out[k] = v
    return out


def evaluation_smoke() -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg_mod.PhysicsSteeringConfig()
    try:
        import torch  # noqa: F401

        pipe = load_sibling(__file__, "pipeline")
        return _round_values(dict(pipe.evaluation_demo(cfg)))
    except ImportError:
        return {"paper": "arXiv:2605.24322", "fallback": True}
