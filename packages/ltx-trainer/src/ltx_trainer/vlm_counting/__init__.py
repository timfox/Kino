"""Visual counting bottleneck in VLMs (arXiv:2605.30170)."""

from ltx_trainer.vlm_counting.config import VLMCountingConfig
from ltx_trainer.vlm_counting.gaps import (
    CountingDiagnostics,
    comparative_match_accuracy,
    regime_for_n,
    synthetic_predicted_count,
)
from ltx_trainer.vlm_counting.go_board import GoBoardSample, patch_embeddings_from_board, sample_go_board
from ltx_trainer.vlm_counting.intervention import mask_top_k_stone_embeddings, steering_accuracy_curve
from ltx_trainer.vlm_counting.ltx_bridge import caption_vision_gap_check, extract_caption_count, ltx_integration_notes
from ltx_trainer.vlm_counting.metrics import (
    fig2_baseline_paradox,
    fig3_gap_curves,
    fig4_comparative_counting,
    fig5_attractor_distribution,
    fig6_circuit_overlap,
    fig7_qwen_baseline,
    fig8_qwen_gaps,
    fig9_layer_probe_qwen,
    fig9_layer_probe_synthetic,
    steering_curve_d11,
    three_stages,
)
from ltx_trainer.vlm_counting.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    paper_limitations,
    training_step_demo,
)
from ltx_trainer.vlm_counting.probing import StonePresenceProbe, train_probe_on_id
from ltx_trainer.vlm_counting.prompts import prompt_bundle

__all__ = [
    "VLMCountingConfig",
    "CountingDiagnostics",
    "GoBoardSample",
    "StonePresenceProbe",
    "benchmark_manifest",
    "caption_vision_gap_check",
    "comparative_match_accuracy",
    "evaluation_demo",
    "extract_caption_count",
    "fig2_baseline_paradox",
    "fig3_gap_curves",
    "fig4_comparative_counting",
    "fig5_attractor_distribution",
    "fig6_circuit_overlap",
    "fig7_qwen_baseline",
    "fig8_qwen_gaps",
    "fig9_layer_probe_qwen",
    "fig9_layer_probe_synthetic",
    "framework_card",
    "ltx_integration_notes",
    "mask_top_k_stone_embeddings",
    "paper_limitations",
    "patch_embeddings_from_board",
    "prompt_bundle",
    "regime_for_n",
    "sample_go_board",
    "steering_accuracy_curve",
    "steering_curve_d11",
    "synthetic_predicted_count",
    "three_stages",
    "train_probe_on_id",
    "training_step_demo",
]
