"""BTS-CAFE — FedDG for stethoscope-induced RSC (arXiv:2605.29862)."""

from ltx_trainer.btscafe.config import BTSCafeConfig
from ltx_trainer.btscafe.devices import device_registry, lodo_splits
from ltx_trainer.btscafe.federated import (
    alignment_penalty,
    fedavg_aggregate,
    global_reference_gradient,
    gradient_alignment_smoke,
    local_loss,
)
from ltx_trainer.btscafe.gin import gin_augment
from ltx_trainer.btscafe.interventions import embedding_intervention_analysis
from ltx_trainer.btscafe.lodo_eval import lodo_eval_smoke, run_all_lodo, run_lodo_fold
from ltx_trainer.btscafe.layout import LIMITATIONS
from ltx_trainer.btscafe.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    figure_ii_embedding_analysis,
    framework_card,
    headline_results,
    pipeline_demo,
    table_ii_main_results,
    table_iii_ablation,
    table_iv_backbone_comparison,
)
from ltx_trainer.btscafe.text_aug import counterfactual_text_augment, neutralize_device

__all__ = [
    "BTSCafeConfig",
    "LIMITATIONS",
    "alignment_penalty",
    "benchmarks_bundle",
    "counterfactual_text_augment",
    "device_registry",
    "embedding_intervention_analysis",
    "evaluation_demo",
    "fedavg_aggregate",
    "figure_ii_embedding_analysis",
    "framework_card",
    "gin_augment",
    "global_reference_gradient",
    "gradient_alignment_smoke",
    "headline_results",
    "local_loss",
    "lodo_eval_smoke",
    "lodo_splits",
    "neutralize_device",
    "pipeline_demo",
    "run_all_lodo",
    "run_lodo_fold",
    "table_ii_main_results",
    "table_iii_ablation",
    "table_iv_backbone_comparison",
]
