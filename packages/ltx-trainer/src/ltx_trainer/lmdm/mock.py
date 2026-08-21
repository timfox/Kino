"""LMDM routing smoke."""

from __future__ import annotations

from typing import Any


def toy_discriminator_scores() -> dict[str, float]:
    return {
        "rollout": 0.55,
        "real": 0.85,
        "matched": 0.12,
        "mismatched": 0.78,
        "fake": 0.25,
    }


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.lmdm.routing import routing_mask

    mask = routing_mask(context_frames=8, target_frames=4)
    scores = toy_discriminator_scores()
    return {"mask_sum": sum(mask), "mask_len": len(mask), **{f"score_{k}": round(v, 3) for k, v in scores.items()}}
