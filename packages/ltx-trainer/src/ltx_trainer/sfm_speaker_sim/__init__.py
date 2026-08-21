"""Speech foundation model vs human speaker similarity (arXiv:2606.05739)."""

from ltx_trainer.sfm_speaker_sim.config import SfmSpeakerSimConfig
from ltx_trainer.sfm_speaker_sim.embedding import (
    cosine_similarity_matrix,
    normalize_scores,
    speaker_embedding,
)
from ltx_trainer.sfm_speaker_sim.eval import eval_smoke, pipeline_demo_export
from ltx_trainer.sfm_speaker_sim.metrics import (
    correspondence_bundle,
    frobenius_distance,
    normalized_laplacian_eigenvalues,
    pearson_lcc,
    spearman_srcc,
    spectral_distance,
)
from ltx_trainer.sfm_speaker_sim.mock import evaluation_smoke
from ltx_trainer.sfm_speaker_sim.models import ModelProfile, model_taxonomy, representative_models
from ltx_trainer.sfm_speaker_sim.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    pipeline_demo,
)
from ltx_trainer.sfm_speaker_sim.regression import table1_regression

__all__ = [
    "ModelProfile",
    "SfmSpeakerSimConfig",
    "annotate_audio_save_data",
    "benchmarks_bundle",
    "correspondence_bundle",
    "cosine_similarity_matrix",
    "eval_smoke",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "frobenius_distance",
    "headline_results",
    "model_taxonomy",
    "normalize_scores",
    "normalized_laplacian_eigenvalues",
    "pearson_lcc",
    "pipeline_demo",
    "pipeline_demo_export",
    "representative_models",
    "speaker_embedding",
    "spearman_srcc",
    "spectral_distance",
    "table1_regression",
]

from ltx_trainer.sfm_speaker_sim.fold import annotate_audio_save_data  # noqa: E402
