"""NEUROK — generative 4D neural object kinematics (arXiv:2605.30347)."""

from ltx_trainer.neurok.active_subspace import active_subspace_basis, lift_latent, project_latent
from ltx_trainer.neurok.config import NeurokConfig
from ltx_trainer.neurok.kinematics import KinematicParameterization, configuration_manifold_dim
from ltx_trainer.neurok.lagrangian import (
    euler_lagrange_step,
    lagrangian,
    simulate_latent_trajectory,
    total_energy,
)
from ltx_trainer.neurok.layout import LIMITATIONS
from ltx_trainer.neurok.metrics import chamfer_l1, chamfer_l2, inverse_kinematics_smoke, voxel_iou
from ltx_trainer.neurok.paper_tables import table_i_inverse_kinematics, table_ii_generative_4d
from ltx_trainer.neurok.pipeline import benchmarks_bundle, evaluation_demo, framework_card, pipeline_demo
from ltx_trainer.neurok.vae import conditional_vae_loss, reparameterize, toy_encoder_decoder_step

__all__ = [
    "LIMITATIONS",
    "KinematicParameterization",
    "NeurokConfig",
    "active_subspace_basis",
    "benchmarks_bundle",
    "chamfer_l1",
    "chamfer_l2",
    "conditional_vae_loss",
    "configuration_manifold_dim",
    "euler_lagrange_step",
    "evaluation_demo",
    "framework_card",
    "inverse_kinematics_smoke",
    "lagrangian",
    "lift_latent",
    "pipeline_demo",
    "project_latent",
    "reparameterize",
    "simulate_latent_trajectory",
    "table_i_inverse_kinematics",
    "table_ii_generative_4d",
    "total_energy",
    "toy_encoder_decoder_step",
    "voxel_iou",
]
