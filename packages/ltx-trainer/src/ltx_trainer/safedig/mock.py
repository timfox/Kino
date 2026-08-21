"""Runnable evaluation smoke for safedig (arXiv stub)."""

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
    cfg = cfg_mod.SafeDIGConfig()
    try:
        import torch  # noqa: F401

        pipe = load_sibling(__file__, "pipeline")
        return _round_values(pipe.evaluation_demo(cfg=cfg))
    except ImportError:
        pass
    return {"paper": "arXiv:2605.30049", "hooks": 3}
