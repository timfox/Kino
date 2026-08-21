"""Runnable evaluation smoke for echo_infinity (arXiv:2606.04527)."""

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
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg_mod.EchoInfinityConfig()
    try:
        import torch  # noqa: F401

        pipe = load_sibling(__file__, "pipeline")
        return _round_values(dict(pipe.evaluation_demo(cfg)))
    except ImportError:
        return _round_values(_numpy_fallback(cfg))


def _numpy_fallback(cfg: Any) -> dict[str, Any]:
    from ltx_trainer.echo_infinity.relative_rope import layout_for_step, verify_layout_in_range

    layout = layout_for_step(50, cfg, has_memory=True)
    return {
        "paper": cfg.paper_arxiv,
        "rope_in_range": verify_layout_in_range(layout, cfg.fmax),
        "memory_frames": cfg.num_memory_query_frames,
    }
