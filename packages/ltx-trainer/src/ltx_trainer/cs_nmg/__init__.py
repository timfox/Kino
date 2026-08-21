"""CS-NMG — POI-aware contrastive CS-ASR (arXiv:2606.06985)."""

from ltx_trainer.cs_nmg.config import CsNmgConfig
from ltx_trainer.cs_nmg.eval import eval_smoke, pipeline_demo
from ltx_trainer.cs_nmg.fold import annotate_audio_save_data
from ltx_trainer.cs_nmg.losses import (
    combined_loss,
    infonce_loss,
    length_normalized_score,
    losses_demo,
    poi_token_weights,
    weighted_ce_loss,
)
from ltx_trainer.cs_nmg.mock import evaluation_smoke
from ltx_trainer.cs_nmg.near_miss import filter_near_miss, near_miss_demo, replace_poi_span
from ltx_trainer.cs_nmg.poi import pier, poi_demo, poi_index_set
from ltx_trainer.cs_nmg.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table2_main_results,
    table3_filter_ablations,
    training_protocol,
)

__all__ = [
    "CsNmgConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "combined_loss",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "filter_near_miss",
    "framework_card",
    "headline_results",
    "infonce_loss",
    "length_normalized_score",
    "losses_demo",
    "near_miss_demo",
    "pier",
    "pipeline_demo",
    "poi_demo",
    "poi_index_set",
    "poi_token_weights",
    "replace_poi_span",
    "table2_main_results",
    "table3_filter_ablations",
    "training_protocol",
    "weighted_ce_loss",
]
