"""SSL features for WikiArt style/genre classification (IRCDL'26)."""

from ltx_trainer.ssl_art.classify import (
    accuracy,
    cosine_similarity,
    knn_zero_shot_predict,
    linear_predict,
    zero_shot_predict,
)
from ltx_trainer.ssl_art.config import SSLArtConfig
from ltx_trainer.ssl_art.layout import LIMITATIONS
from ltx_trainer.ssl_art.pipeline import (
    benchmarks_bundle,
    best_clip_linear_metrics,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_i_wikiart_results,
)
from ltx_trainer.ssl_art.prompts import build_label_prompts, genre_prompt, style_prompt
from ltx_trainer.ssl_art.retrieval import top_k_retrieve

__all__ = [
    "LIMITATIONS",
    "SSLArtConfig",
    "accuracy",
    "benchmarks_bundle",
    "best_clip_linear_metrics",
    "build_label_prompts",
    "cosine_similarity",
    "evaluation_demo",
    "framework_card",
    "genre_prompt",
    "knn_zero_shot_predict",
    "linear_predict",
    "pipeline_demo",
    "style_prompt",
    "table_i_wikiart_results",
    "top_k_retrieve",
    "zero_shot_predict",
]
