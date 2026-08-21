"""Runnable evaluation smoke for before_shutter (arXiv stub)."""

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
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [_round_values(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def evaluation_smoke() -> dict[str, Any]:
    cfg_mod = load_sibling(__file__, "config")
    cfg = cfg_mod.BeforeShutterConfig()
    try:
        import torch  # noqa: F401

        pipe = load_sibling(__file__, "pipeline")
        return _round_values(pipe.evaluation_demo(cfg=cfg))
    except ImportError:
        pass
    graph_mod = load_sibling(__file__, "scene_graph")
    g = graph_mod.build_demo_graph()
    return {
        "paper": "arXiv:2605.30318",
        "graph_nodes": len(g.nodes_non) + len(g.nodes_emi),
    }
