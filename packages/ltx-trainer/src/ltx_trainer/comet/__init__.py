"""COMET — CLAP concept-space dissection via PLS-SVD (arXiv:2605.29628)."""

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.gap import cosine_sim, embedding_shift, modality_gap_sources
from ltx_trainer.comet.layout import LIMITATIONS
from ltx_trainer.comet.pls import (
    center_embeddings,
    covariance_decomposition,
    net_useful_contribution,
    pls_svd,
    project_coefficients,
    reconstruct_from_projections,
    similarity_direct_cross,
)
from ltx_trainer.comet.plshead import head_energy_ratio, linear_projection_decoding, plshead_truncate
from ltx_trainer.comet.retrieval import retrieval_smoke, text_to_audio_retrieval
from ltx_trainer.comet.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
    table_i_contribution,
    table_iii_retrieval_audiocaps,
    table_iii_retrieval_clotho,
    table_iv_projection_decoding,
    table_v_captioning_audiocaps,
    table_v_captioning_clotho,
)

__all__ = [
    "CometConfig",
    "LIMITATIONS",
    "benchmarks_bundle",
    "center_embeddings",
    "cosine_sim",
    "covariance_decomposition",
    "embedding_shift",
    "evaluation_demo",
    "framework_card",
    "head_energy_ratio",
    "headline_results",
    "linear_projection_decoding",
    "modality_gap_sources",
    "net_useful_contribution",
    "pipeline_demo",
    "pls_svd",
    "plshead_truncate",
    "project_coefficients",
    "reconstruct_from_projections",
    "retrieval_smoke",
    "similarity_direct_cross",
    "table_i_contribution",
    "table_iii_retrieval_audiocaps",
    "table_iii_retrieval_clotho",
    "table_iv_projection_decoding",
    "table_v_captioning_audiocaps",
    "table_v_captioning_clotho",
    "text_to_audio_retrieval",
]
