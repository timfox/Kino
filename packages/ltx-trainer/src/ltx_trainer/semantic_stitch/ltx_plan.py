"""LTX / AV-fold plan for SemanticStitch × proceduralsky ERP training."""

from __future__ import annotations

from typing import Any

from ltx_trainer.semantic_stitch.config import PAPER_ARXIV
from ltx_trainer.semantic_stitch.integration import (
    DEFAULT_SKIES_PROJECT,
    proceduralsky_card,
    proceduralsky_pipeline_plan,
)


def ltx_training_plan(
    *,
    skies_project: str = DEFAULT_SKIES_PROJECT,
    use_semantic_stitch_weights: bool = True,
    min_semantic_stitch_weight: float = 0.12,
) -> dict[str, Any]:
    """Recommend fold hook + trainer flags for salient-aware stitched ERP skies."""
    return {
        "paper": PAPER_ARXIV,
        "skies_project": skies_project,
        "preprocessed_roots": {
            "skies_merged": f"${{WORK_ROOT}}/precomputed/merged_{skies_project}",
            "stitched_erp": f"${{WORK_ROOT}}/precomputed/{skies_project}/stitched_erp",
        },
        "trainer_flags": {
            "use_semantic_stitch_weights": use_semantic_stitch_weights,
            "min_semantic_stitch_weight": min_semantic_stitch_weight,
            "note": "Upweight shards with high seam_coherence after SemanticStitch merge",
        },
        "fold_hook": "ltx_trainer.semantic_stitch.fold.annotate_video_latent_data",
        "meta_hints": {
            "semantic_stitch_merged": True,
            "salient_seam_clear": True,
            "overlap_captures": 2,
        },
        "proceduralsky": proceduralsky_card(),
        "pipeline": proceduralsky_pipeline_plan(skies_project=skies_project),
    }


def gopex_env_exports(skies_project: str = DEFAULT_SKIES_PROJECT) -> dict[str, str]:
    return {
        "GOPEX_SKIES_PROJECT": skies_project,
        "GOPEX_AV_FOLD_HOOKS": "semantic_stitch,pantheon360,sphere360",
        "GOPEX_ENABLE_AV_FOLD": "1",
    }
