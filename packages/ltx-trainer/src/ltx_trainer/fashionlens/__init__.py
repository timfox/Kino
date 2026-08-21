"""FashionLens — versatile fashion image retrieval via task-adaptive learning (arXiv:2605.22552)."""

from ltx_trainer.fashionlens.config import FashionLensConfig
from ltx_trainer.fashionlens.ggas import (
    argmax_task_index,
    ema_difficulty,
    retrieval_token_difficulty,
    sampling_probabilities,
    sampling_score,
)
from ltx_trainer.fashionlens.layout import LIMITATIONS
from ltx_trainer.fashionlens.losses import infonce_retrieval_loss, mean_reciprocal_rank_from_r_at, total_training_loss
from ltx_trainer.fashionlens.pgsqc import (
    adaptation_proposal,
    frobenius_squared,
    interpolation_lambda,
    orthogonality_loss_frobenius,
    slerp,
)
from ltx_trainer.fashionlens.pipeline import (
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_ii_main_results,
    table_iii_ablation,
)
from ltx_trainer.fashionlens.ufire import ufire_scale_summary, ufire_task_rubric

__all__ = [
    "LIMITATIONS",
    "FashionLensConfig",
    "adaptation_proposal",
    "argmax_task_index",
    "ema_difficulty",
    "evaluation_demo",
    "framework_card",
    "frobenius_squared",
    "infonce_retrieval_loss",
    "interpolation_lambda",
    "mean_reciprocal_rank_from_r_at",
    "orthogonality_loss_frobenius",
    "pipeline_demo",
    "retrieval_token_difficulty",
    "sampling_probabilities",
    "sampling_score",
    "slerp",
    "table_ii_main_results",
    "table_iii_ablation",
    "total_training_loss",
    "ufire_scale_summary",
    "ufire_task_rubric",
]
