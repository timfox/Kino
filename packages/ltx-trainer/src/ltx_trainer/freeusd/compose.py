"""Spatial LLM stub: bible/slate → USDA → LTX spatial lock (CID train target)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.freeusd.config import FreeUSDConfig
from ltx_trainer.freeusd.usda import parse_blocking, shot_to_usda, usda_to_spatial_lock


def compose_shot(
    shot: dict[str, Any],
    *,
    cfg: FreeUSDConfig | None = None,
) -> dict[str, Any]:
    """Deterministic spatial compose. Later: swap in a CID-trained USD text LLM."""
    cfg = cfg or FreeUSDConfig()
    beat = str((shot.get("blocking") or {}).get("action") or shot.get("beat") or "")
    poses = parse_blocking(beat)
    usda = shot_to_usda(shot, cfg=cfg, poses=poses)
    lock = usda_to_spatial_lock(usda)
    prompt = str(shot.get("prompt") or shot.get("text") or "")
    composed = f"{lock} {prompt}".strip() if prompt else lock
    stem = shot.get("tv_stem") or (shot.get("files") or {}).get("tv_stem")
    return {
        "slate": shot.get("slate_label") or shot.get("slate"),
        "tv_stem": stem,
        "usda": usda,
        "spatial_lock": lock,
        "composed_prompt": composed,
        "cast_marks": [
            {"name": p.name, "prim": p.prim, "translate": list(p.translate), "pose": p.pose}
            for p in poses
        ],
        "backend": "rule_spatial_llm",
        "train_target": "cid_freeusd_text_llm",
    }


def compose_season(shots: list[dict[str, Any]], *, cfg: FreeUSDConfig | None = None) -> dict[str, Any]:
    layers = [compose_shot(s, cfg=cfg) for s in shots]
    return {
        "n_shots": len(layers),
        "stems": [x.get("tv_stem") for x in layers],
        "shots": layers,
    }
