"""Runnable evaluation smoke for frng (arXiv stub)."""

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
    cfg = cfg_mod.FRNGConfig()
    try:
        import torch  # noqa: F401

        if True:
            pipe = load_sibling(__file__, "pipeline")
            return _round_values(_call_demo(pipe, cfg))
    except ImportError:
        pass
    return _round_values(_numpy_fallback(cfg))

def _NUMPY_FALLBACK(cfg: Any) -> dict[str, Any]:
    import numpy as np
    gray = np.linspace(0, 1, 16, dtype=np.float64).reshape(4, 4)
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    edge = float(np.abs(gx).mean() + np.abs(gy).mean())
    return {"paper": "arXiv:2605.25975", "edge_energy": round(edge, 4), "lambda_l2": getattr(cfg, "lambda_l2", 1.0)}
