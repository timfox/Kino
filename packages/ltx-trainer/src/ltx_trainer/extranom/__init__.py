"""ExtrAnom: women-centric multi-modal VAD dataset (Sangeeta et al., arXiv:2605.25806).

Reference: category statistics, caption metric proxies, paper Tables 1–4, Holmes-VAU error rates.
Full dataset videos and LLM annotation pipeline are external — see https://github.com/A24CS09005/ExtrAnom
"""

from ltx_trainer.extranom.categories import (
    anomaly_category_counts,
    category_share_of_anomalous,
    category_share_of_total,
    table_category_statistics,
)
from ltx_trainer.extranom.config import ExtrAnomConfig
from ltx_trainer.extranom.metrics import caption_metrics, mean_caption_metrics, rouge_l_recall, unigram_f1
from ltx_trainer.extranom.pipeline import (
    evaluation_demo,
    example_ground_truth_chain_snatching,
    framework_card,
    holmes_vau_error_analysis,
    table_dataset_comparison,
    table_vision_auc,
    table_vlm_caption_similarity,
    training_step_demo,
)

__all__ = [
    "ExtrAnomConfig",
    "anomaly_category_counts",
    "caption_metrics",
    "category_share_of_anomalous",
    "category_share_of_total",
    "evaluation_demo",
    "example_ground_truth_chain_snatching",
    "framework_card",
    "holmes_vau_error_analysis",
    "mean_caption_metrics",
    "rouge_l_recall",
    "table_category_statistics",
    "table_dataset_comparison",
    "table_vision_auc",
    "table_vlm_caption_similarity",
    "training_step_demo",
    "unigram_f1",
]
