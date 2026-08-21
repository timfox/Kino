"""OpenScene semantic → ITU-R P.2040 materials (§IV-D)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig

# Semantic category → electromagnetic material (paper §IV-D)
SEMANTIC_TO_EM: dict[str, str] = {
    "wall": "concrete",
    "floor": "concrete",
    "ceiling": "concrete",
    "window": "glass",
    "door": "wood",  # wooden doors per paper
    "wooden_furniture": "wood",
    "metal_cabinet": "metal",
    "glass_door": "glass",
}


def map_semantic_labels(labels: list[str]) -> dict[str, str]:
    return {lab: SEMANTIC_TO_EM.get(lab, "concrete") for lab in labels}


def material_binding_summary(cfg: RfdtChannelConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RfdtChannelConfig()
    mapping = map_semantic_labels(list(cfg.openscene_categories))
    return {
        "openscene_role": "semantic_region_cues_only",
        "itu_reference": cfg.itu_material_ref,
        "mapping": mapping,
        "multi_material_set": list(cfg.semantic_materials),
        "all_concrete_baseline": "uniform concrete on all faces",
    }
