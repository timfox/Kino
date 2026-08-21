"""LTX / AV-fold hooks for Pantheon360 × proceduralsky 360° HDR training."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pantheon360.config import Pantheon360Config
from ltx_trainer.pantheon360.integration import (
    DEFAULT_SKIES_PROJECT,
    proceduralsky_card,
    proceduralsky_pipeline_plan,
)


def ltx_training_plan(
    *,
    skies_project: str = DEFAULT_SKIES_PROJECT,
    use_pantheon360_weights: bool = True,
    min_pantheon360_weight: float = 0.15,
) -> dict[str, Any]:
    """Recommend AV-fold / trainer flags when mixing proceduralsky skies with 360° scene clips."""
    cfg = Pantheon360Config()
    return {
        "paper": cfg.paper_arxiv,
        "skies_project": skies_project,
        "preprocessed_roots": {
            "skies_merged": f"${{WORK_ROOT}}/precomputed/merged_{skies_project}",
            "skies_hdr": f"${{PRECOMPUTED_HDR_ROOT}}/{skies_project}",
        },
        "video_dims": [cfg.erp_width, cfg.erp_height, cfg.num_frames],
        "trainer_flags": {
            "use_pantheon360_weights": use_pantheon360_weights,
            "min_pantheon360_weight": min_pantheon360_weight,
            "finetune_text_connectors": False,
            "note": "Upweight shards where fold annotator reports high cache_fusion_readiness",
        },
        "fold_hooks": [
            "ltx_trainer.semantic_stitch.fold.annotate_video_latent_data",
            "ltx_trainer.pantheon360.fold.annotate_video_latent_data",
        ],
        "fold_hook": "ltx_trainer.pantheon360.fold.annotate_video_latent_data",
        "caption_hint": (
            "2:1 equirectangular 360 clip; mention horizon, cloud motion, seamless wrap, "
            "and trajectory-consistent geometry when visible."
        ),
        "proceduralsky": proceduralsky_card(),
        "pipeline": proceduralsky_pipeline_plan(skies_project=skies_project),
    }


def gopex_env_exports(skies_project: str = DEFAULT_SKIES_PROJECT) -> dict[str, str]:
    """Shell-friendly env for proceduralsky + Pantheon360 joint prep."""
    return {
        "GOPEX_SKIES_PROJECT": skies_project,
        "GOPEX_SKIES_RESOLUTION_BUCKETS": "1024x512x41",
        "GOPEX_PREP_PROJECTS": skies_project,
        "HDR_VAE_ENCODING": "logc3",
    }
