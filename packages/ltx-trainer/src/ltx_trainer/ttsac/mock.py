"""TT-SAC toy generator + evaluation smoke (arXiv:2605.25488)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.smoke_util import load_sibling


class ToyTalkingHead(nn.Module):
    """Tiny identity encoder / frame generator for pipeline smoke tests."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def encode(self, frame: Tensor) -> Tensor:
        return self.proj(frame)

    def generate(self, f: Tensor, audio: Any = None, t: int = 0) -> Tensor:
        if isinstance(audio, int):
            t = int(audio)
        # Identity frame synthesis so MC refinement is a stable fixed point.
        return f


def toy_compose(f: Tensor, model: ToyTalkingHead, t: int) -> Tensor:
    """One (E ∘ G)(f, ·) step for Monte Carlo stacks."""
    frame = model.generate(f, t=t)
    return model.encode(frame)


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
    cfg = cfg_mod.TTSACConfig()
    try:
        import torch  # noqa: F401

        if True:
            pipe = load_sibling(__file__, "pipeline")
            return _round_values(_call_demo(pipe, cfg))
    except ImportError:
        pass
    return _round_values(_numpy_fallback(cfg))

def _NUMPY_FALLBACK(cfg: Any) -> dict[str, Any]:
    sigma2, k = 0.04, 8
    var_agg = sigma2 / k
    best_k, best_obj = 1, float("inf")
    for trial_k in range(1, 11):
        obj = sigma2 / trial_k + (0.02 * trial_k) ** 2
        if obj < best_obj:
            best_obj, best_k = obj, trial_k
    return {"paper": "arXiv:2605.25488", "iid_var_at_k8": round(var_agg, 6), "optimal_k": best_k}
