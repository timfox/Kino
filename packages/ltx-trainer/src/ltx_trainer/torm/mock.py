"""TORM latent reasoning smoke (arXiv:2605.26014)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").TORMConfig()
    try:
        import torch  # noqa: F401

        evaluation_demo = load_sibling(__file__, "pipeline").evaluation_demo

        out = evaluation_demo(device="cpu")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    except ImportError:
        return {
            "paper": "arXiv:2605.26014",
            "num_latent_slots": float(cfg.num_latent_slots),
            "torm_videomme_stub": 61.0,
        }
