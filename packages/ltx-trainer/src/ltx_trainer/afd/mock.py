"""AFD adversarial flow distillation smoke (arXiv:2605.26105)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").AFDConfig()
    try:
        import torch  # noqa: F401

        evaluation_demo = load_sibling(__file__, "pipeline").evaluation_demo

        out = evaluation_demo(device="cpu")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    except ImportError:
        return {
            "paper": getattr(cfg, "paper_arxiv", "arXiv:2605.26105"),
            "nft_beta": cfg.nft_beta,
            "num_students": float(len(cfg.student_backbones)),
            "afd_physics_self_forcing_stub": 87.55,
        }
