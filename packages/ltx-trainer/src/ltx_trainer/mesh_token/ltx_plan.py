"""LTX / Kino hooks for render-free mesh motion control."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mesh_token.config import MeshTokenConfig


def ltx_integration_plan(cfg: MeshTokenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MeshTokenConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "Character I2V with SMPL/GVHMR mesh sidecars instead of pose-map video",
            "Disentangled trajectory vs body-pose editing for teleplay shot fixes",
            "Per-latent-frame motion cross-attention aligned to LTX temporal stride",
        ],
        "note": "Reference stubs; full Wan-2.1 finetune not bundled",
    }
