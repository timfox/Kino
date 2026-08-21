"""SVHighlights — extremely long sport video highlight detection (arXiv:2606.06926)."""

from ltx_trainer.svhighlights.alignment import (
    align_highlight_sequence,
    alignment_stats,
    best_match_frame,
    downsample_frame,
    psnr,
)
from ltx_trainer.svhighlights.config import SportCategory, SvHighlightsConfig
from ltx_trainer.svhighlights.labeling import clip_labels, highlight_intervals_from_indices, labels_from_alignment
from ltx_trainer.svhighlights.layout import LIMITATIONS
from ltx_trainer.svhighlights.ltx_plan import gopex_env_snippet, ltx_integration_plan
from ltx_trainer.svhighlights.metrics import (
    average_precision,
    evaluate_predictions,
    hit_at_k,
    hit_at_one,
    temporal_iou,
    window_f1,
)
from ltx_trainer.svhighlights.mock import (
    evaluation_smoke,
    run_alignment_smoke,
    run_label_smoke,
    run_metrics_smoke,
    run_tf_selector_smoke,
)
from ltx_trainer.svhighlights.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    table_i_benchmarks,
    table_iii_alignment_quality,
    table_ix_llm_ablation,
    table_vii_main_results,
    table_viii_vlm_ablation,
    table_x_modality_ablation,
    table_xii_window_f1,
    table_tau_ablation,
    trimming_cues,
)
from ltx_trainer.svhighlights.segmentation import Segment, WordSpan, merge_shots_by_transcript, segment_video_stub
from ltx_trainer.svhighlights.tf_selector import (
    build_segment_features,
    clip_scores_from_segments,
    llm_saliency_stub,
    run_tf_selector_stub,
)

__all__ = [
    "LIMITATIONS",
    "Segment",
    "SportCategory",
    "SvHighlightsConfig",
    "WordSpan",
    "align_highlight_sequence",
    "alignment_stats",
    "average_precision",
    "benchmarks_bundle",
    "best_match_frame",
    "build_segment_features",
    "clip_labels",
    "clip_scores_from_segments",
    "downsample_frame",
    "evaluate_predictions",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "gopex_env_snippet",
    "highlight_intervals_from_indices",
    "hit_at_k",
    "hit_at_one",
    "labels_from_alignment",
    "llm_saliency_stub",
    "ltx_integration_plan",
    "merge_shots_by_transcript",
    "psnr",
    "run_alignment_smoke",
    "run_label_smoke",
    "run_metrics_smoke",
    "run_tf_selector_stub",
    "run_tf_selector_smoke",
    "segment_video_stub",
    "table_i_benchmarks",
    "table_iii_alignment_quality",
    "table_ix_llm_ablation",
    "table_vii_main_results",
    "table_viii_vlm_ablation",
    "table_x_modality_ablation",
    "table_xii_window_f1",
    "table_tau_ablation",
    "temporal_iou",
    "trimming_cues",
    "window_f1",
]
