"""GOPEX stack integration for Mirage latent spatial memory (arXiv:2606.09828)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.mirage.config import MirageConfig


def gopex_stack_card() -> dict[str, Any]:
    cfg = MirageConfig()
    return {
        "paper": cfg.paper_arxiv,
        "website": cfg.website,
        "mirage_role": (
            "Latent 3D cache M={(p_i,f_i)} with depth back-projection and z-buffer readout; "
            "ControlNet-style side branch on Wan2.2 TI2V for world-model rollouts."
        ),
        "scripts": {
            "gopex": "./scripts/gopex-mirage.sh",
            "kino": "./scripts/kino-mirage.sh",
        },
        "fold_hook": "mirage",
        "readiness_field": "readout_coverage",
        "trainer_flag": "use_mirage_weights",
    }


def native_evolve_plan() -> dict[str, Any]:
    """Suggested steps when mixing Mirage sidecars into merged_native training."""
    return {
        "preprocess": [
            {
                "id": "mirage_backfill",
                "command": "GOPEX_AV_FOLD_HOOKS=mirage ./scripts/kino-av-fold-native.sh backfill --merge",
                "artifact": "meta.mirage.readout_coverage on latent shards",
            },
            {
                "id": "mirage_smoke",
                "command": "./scripts/kino-mirage.sh smoke",
            },
        ],
        "train": [
            {
                "id": "mirage_weights",
                "note": "Set av_fold.use_mirage_weights=true; upweight clips with high readout_coverage",
            },
            {
                "id": "world_model_profile",
                "command": "eval \"$(python tools/ltx_research_profile.py env world_model)\"",
            },
        ],
    }
