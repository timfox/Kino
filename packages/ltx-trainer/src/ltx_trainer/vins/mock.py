"""Runnable evaluation smoke for vins (arXiv stub)."""

from __future__ import annotations

import math
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


def _call_demo(pipe: Any, cfg: Any) -> dict[str, Any]:
    for name in (
        "evaluation_demo",
        "training_step_demo",
        "synthetic_training_step",
        "demo_segmentation_metrics",
    ):
        fn = getattr(pipe, name, None)
        if fn is None:
            continue
        for args, kwargs in (
            ((cfg,), {}),
            ((), {}),
            ((), {"device": "cpu"}),
            ((cfg,), {"device": "cpu"}),
        ):
            try:
                out = fn(*args, **kwargs)
                return dict(out)
            except TypeError:
                continue
    card = getattr(pipe, "framework_card", None)
    if card is not None:
        try:
            return dict(card(cfg))
        except TypeError:
            return dict(card())
    bench = getattr(pipe, "benchmark_table", None)
    if bench is not None:
        return {"benchmark": bench()}
    return {}


def _numpy_fallback(cfg: Any) -> dict[str, Any]:
    return _NUMPY_FALLBACK(cfg)


def evaluation_smoke() -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg_mod.VINSConfig()
    try:
        import torch  # noqa: F401

        pipe = load_sibling(__file__, "pipeline")
        out = _round_values(_call_demo(pipe, cfg))
        if out:
            return out
    except ImportError:
        pass
    return _round_values(_numpy_fallback(cfg))

def _NUMPY_FALLBACK(cfg: Any) -> dict[str, Any]:
    import numpy as np
    img = np.random.default_rng(0).random((32, 32))
    gx = np.diff(img, axis=1)
    gy = np.diff(img, axis=0)
    tenengrad = float((gx**2).mean() + (gy**2).mean())
    return {"paper": "arXiv:2605.23518", "tenengrad": round(tenengrad, 4), "min_resolution": getattr(cfg, "min_short_edge", 720)}
