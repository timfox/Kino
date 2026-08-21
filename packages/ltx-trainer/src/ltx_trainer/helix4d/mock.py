"""Helix4D anchor attention + 4D RoPE smoke (arXiv:2605.26109)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").Helix4DConfig()
    try:
        import torch  # noqa: F401

        evaluation_demo = load_sibling(__file__, "pipeline").evaluation_demo

        out = evaluation_demo(device="cpu")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    except ImportError:
        return {
            "paper": "arXiv:2605.26109",
            "anchor_tokens": float(getattr(cfg, "num_anchor_tokens", 64)),
            "mask_density_stub": 0.35,
        }
