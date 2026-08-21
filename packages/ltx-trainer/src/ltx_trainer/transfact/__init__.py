"""TransFACT — bovine embryo transferability from time-lapse video (arXiv:2605.18923)."""

from ltx_trainer.transfact.config import TransfactConfig
from ltx_trainer.transfact.layout import ARCHITECTURE_NOTES, LIMITATIONS, STAGE_CLASSES
from ltx_trainer.transfact.losses import total_transfact_loss
from ltx_trainer.transfact.mhi import mhi_from_sequence, motion_mask, update_mhi
from ltx_trainer.transfact.stages import cell_count_to_stage, stage_frame_labels
from ltx_trainer.transfact.mock import evaluation_smoke, mhi_demo
from ltx_trainer.transfact.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.transfact.tables import (
    progressive_accuracy_curve,
    table1_input_modalities,
    table2_vs_sfr,
)

__all__ = [
    "ARCHITECTURE_NOTES",
    "LIMITATIONS",
    "STAGE_CLASSES",
    "TransfactConfig",
    "benchmarks_bundle",
    "cell_count_to_stage",
    "evaluation_demo",
    "stage_frame_labels",
    "total_transfact_loss",
    "evaluation_smoke",
    "framework_card",
    "mhi_demo",
    "mhi_from_sequence",
    "motion_mask",
    "progressive_accuracy_curve",
    "table1_input_modalities",
    "table2_vs_sfr",
    "update_mhi",
]
