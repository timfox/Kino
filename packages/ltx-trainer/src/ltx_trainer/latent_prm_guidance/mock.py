"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.paper import paper_card
from ltx_trainer.latent_prm_guidance.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "prm_beats_latent_unguided": demo["prm_beats_latent_unguided"],
        "status": "ok" if demo["prm_beats_latent_unguided"] else "fail",
    }
