"""TADA — JPEG steganalysis CSM mitigation via data adaptation (arXiv:2605.21523)."""

from ltx_trainer.tada.config import TADAConfig
from ltx_trainer.tada.emulator import (
    apply_emulator,
    project_to_symmetric_sum_one,
    symmetric_kernel_from_params,
)
from ltx_trainer.tada.layout import LIMITATIONS
from ltx_trainer.tada.loss import regret_stub, tada_loss
from ltx_trainer.tada.pipeline import (
    benchmarks_bundle,
    csm_formalization,
    evaluation_demo,
    framework_card,
    pipeline_demo,
    table_i_toy_kernels_detailed,
    table_ii_flickr_targets,
    table_iii_target_regrets,
)
from ltx_trainer.tada.residuals import KB_KERNEL, kb_residual, patch_residuals, residual_covariance

__all__ = [
    "KB_KERNEL",
    "LIMITATIONS",
    "TADAConfig",
    "apply_emulator",
    "benchmarks_bundle",
    "csm_formalization",
    "evaluation_demo",
    "framework_card",
    "kb_residual",
    "patch_residuals",
    "pipeline_demo",
    "project_to_symmetric_sum_one",
    "regret_stub",
    "residual_covariance",
    "symmetric_kernel_from_params",
    "table_i_toy_kernels_detailed",
    "table_ii_flickr_targets",
    "table_iii_target_regrets",
    "tada_loss",
]
