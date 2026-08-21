"""Automatic contextual audio denoising stub (arXiv:2605.22262)."""

from ltx_trainer.acad.config import AcadConfig
from ltx_trainer.acad.layout import LIMITATIONS
from ltx_trainer.acad.losses import asc_cross_entropy_loss, joint_loss, si_snr_db
from ltx_trainer.acad.masking import contextual_denoising_mask, masked_magnitude
from ltx_trainer.acad.mock import evaluation_smoke
from ltx_trainer.acad.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    headline_results,
    table1_model_setups,
    table2_metrics,
)

__all__ = [
    "LIMITATIONS",
    "AcadConfig",
    "asc_cross_entropy_loss",
    "benchmarks_bundle",
    "contextual_denoising_mask",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "headline_results",
    "joint_loss",
    "masked_magnitude",
    "si_snr_db",
    "table1_model_setups",
    "table2_metrics",
]
