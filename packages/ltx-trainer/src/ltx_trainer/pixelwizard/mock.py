"""PixelWizard anchor + shortcut smoke (arXiv:2605.25801)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg: Any | None = None) -> dict[str, Any]:
    cfg = cfg or load_sibling(__file__, "config").PixelWizardConfig()
    out: dict[str, Any] = {
        "paper": "arXiv:2605.25801",
        "shortcut_candidates_k": float(cfg.shortcut_candidates_k),
        "hr_inference_steps": float(cfg.hr_inference_steps),
    }

    try:
        import torch

        shortcut = load_sibling(__file__, "shortcut")
        pipe = load_sibling(__file__, "pipeline")

        steps = shortcut.candidate_step_sizes(500, k=cfg.shortcut_candidates_k)
        chosen = shortcut.select_shortcut_step(500, k=cfg.shortcut_candidates_k)
        out["shortcut_step_chosen"] = float(chosen)
        out["shortcut_candidates"] = len(steps)
        demo = pipe.evaluation_demo()
        out.update({k: v for k, v in demo.items() if isinstance(v, (int, float, str, bool))})
    except ImportError:
        out["shortcut_step_chosen"] = float(cfg.shortcut_candidates_k)

    return out
