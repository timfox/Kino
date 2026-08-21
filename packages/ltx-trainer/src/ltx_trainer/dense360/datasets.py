"""Dense360 dataset and Dense360-Bench card (Sec. 3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dense360.config import (
    BENCH_ENTITIES,
    BENCH_ERP_IMAGES,
    ENTITY_CAPTIONS,
    PANORAMA_COUNT,
    REFERRING_EXPRESSIONS,
    SCENE_DESCRIPTIONS,
)


def datasets_card() -> dict[str, Any]:
    return {
        "Dense360": {
            "panoramas": PANORAMA_COUNT,
            "entity_captions": ENTITY_CAPTIONS,
            "referring_expressions": REFERRING_EXPRESSIONS,
            "entity_grounded_scene_descriptions": SCENE_DESCRIPTIONS,
            "scene_split": {"indoor_pct": 32.74, "outdoor_pct": 67.26},
            "entity_quadrants_pct": {
                "front": 40.55,
                "right": 21.97,
                "back": 20.25,
                "left": 11.36,
            },
            "pipeline_levels": ["L1 entity masks", "L2 captions+reliability", "L3 scene descriptions"],
        },
        "Dense360-Bench": {
            "erp_images": BENCH_ERP_IMAGES,
            "entities": BENCH_ENTITIES,
            "grounding_questions": BENCH_ENTITIES,
            "captioning_questions": BENCH_ENTITIES,
            "metrics": {"grounding": "mask IoU", "captioning": "key-phrase recall (GPT-4o judge)"},
        },
        "training": "30% LLaVA SFT + full Dense360; Qwen2.5-VL-3B + LoRA",
    }
