"""Speech quality embeddings for local degradation detection (arXiv:2605.21332)."""

from ltx_trainer.speech_quality_emb.config import SpeechQualityEmbConfig
from ltx_trainer.speech_quality_emb.detection import cosine_similarity_to_enrollment, detect_by_threshold
from ltx_trainer.speech_quality_emb.layout import LIMITATIONS
from ltx_trainer.speech_quality_emb.losses import (
    frame_pseudo_l1,
    supervised_contrastive_stub,
    total_training_loss,
    utterance_mae_loss,
)
from ltx_trainer.speech_quality_emb.mixup import mix_waveforms_stub, partial_mixup_waveform_mask, pseudo_frame_scores
from ltx_trainer.speech_quality_emb.mock import evaluation_smoke
from ltx_trainer.speech_quality_emb.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_training_setup,
    table2_nisqa_test_sim_partial_mixup,
    table3_libri_augmented_test_clean,
    table4_verification_retrieval_nisqa,
    table5_libri_verification_retrieval,
    table6_joint_clustering,
)

__all__ = [
    "LIMITATIONS",
    "SpeechQualityEmbConfig",
    "benchmarks_bundle",
    "cosine_similarity_to_enrollment",
    "detect_by_threshold",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "frame_pseudo_l1",
    "headline_results",
    "mix_waveforms_stub",
    "partial_mixup_waveform_mask",
    "pseudo_frame_scores",
    "supervised_contrastive_stub",
    "table1_training_setup",
    "table2_nisqa_test_sim_partial_mixup",
    "table3_libri_augmented_test_clean",
    "table4_verification_retrieval_nisqa",
    "table5_libri_verification_retrieval",
    "table6_joint_clustering",
    "total_training_loss",
    "utterance_mae_loss",
]
