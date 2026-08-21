"""DIVA toy flows + evaluation smoke (arXiv:2605.25328)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.smoke_util import load_sibling


@dataclass
class AnchorSample:
  """Paired understanding / generation hidden states on one anchor."""

  h_u: Tensor
  h_g: Tensor


def make_toy_flows(*, batch: int, hidden_dim: int, seed: int = 0) -> list[AnchorSample]:
    gen = torch.Generator().manual_seed(seed)
    samples: list[AnchorSample] = []
    for _ in range(batch):
        h_u = torch.randn(hidden_dim, generator=gen)
        h_g = h_u + 0.25 * torch.randn(hidden_dim, generator=gen)
        samples.append(AnchorSample(h_u=h_u, h_g=h_g))
    return samples


def random_mask_ratio(
    *,
    low: float = 0.2,
    high: float = 0.6,
    generator: torch.Generator | None = None,
) -> float:
    if generator is not None:
        return float(low + (high - low) * torch.rand((), generator=generator).item())
    return low + (high - low) * 0.4


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
    cfg = cfg_mod.DIVAConfig()
    try:
        import torch  # noqa: F401

        if True:
            pipe = load_sibling(__file__, "pipeline")
            return _round_values(_call_demo(pipe, cfg))
    except ImportError:
        pass
    return _round_values(_numpy_fallback(cfg))

def _NUMPY_FALLBACK(cfg: Any) -> dict[str, Any]:
    import random
    rng = random.Random(42)
    return {
        "paper": "arXiv:2605.25328",
        "batch_size": 4,
        "hidden_dim": 16,
        "mask_ratio": round(0.2 + 0.4 * rng.random(), 4),
        "torch_available": False,
    }
