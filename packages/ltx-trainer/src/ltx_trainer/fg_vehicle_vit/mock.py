"""Runnable evaluation smoke for fg_vehicle_vit (arXiv:2606.05149)."""

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
        else:
            out[k] = v
    return out


def evaluation_smoke() -> dict[str, Any]:
    pipe = load_sibling(__file__, "pipeline")
    return _round_values(dict(pipe.evaluation_demo()))
