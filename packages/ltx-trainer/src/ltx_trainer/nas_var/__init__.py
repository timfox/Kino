"""NAS-VAR — next-acceleration-scale autoregressive MRI reconstruction (arXiv:2605.19354)."""

from ltx_trainer.nas_var.config import NasVarConfig
from ltx_trainer.nas_var.layout import ARCHITECTURE_NOTES, BASELINES, LIMITATIONS, SCALE_CHAIN
from ltx_trainer.nas_var.distill import privileged_distillation_loss, reverse_kl
from ltx_trainer.nas_var.mock import evaluation_smoke, scale_factorization_demo
from ltx_trainer.nas_var.rollout import rollout_next_scales
from ltx_trainer.nas_var.pipeline import benchmarks_bundle, evaluation_demo, framework_card
from ltx_trainer.nas_var.tables import table1_cartesian_x, table2_radial, table3_cartesian_y

__all__ = [
    "ARCHITECTURE_NOTES",
    "BASELINES",
    "LIMITATIONS",
    "SCALE_CHAIN",
    "NasVarConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "privileged_distillation_loss",
    "reverse_kl",
    "rollout_next_scales",
    "scale_factorization_demo",
    "table1_cartesian_x",
    "table2_radial",
    "table3_cartesian_y",
]
