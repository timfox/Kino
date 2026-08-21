"""R5DGS: semantic-aware physics-informed 4D Gaussians with rigid inference (Gridusov et al., arXiv:2605.25909).

Reference building blocks: identity compositing (Eq. 1), training losses (Eq. 2, 6, 7), rigid-body
extrapolation (Eq. 3–5), and CLIP-style embedding lookup (Eq. 8). **TRACE** / full TRD integrator and
differentiable 3DGS rasterizer are **not** included — integrate with your own dynamics and renderer.
"""

from ltx_trainer.r5dgs.config import R5DGSConfig
from ltx_trainer.r5dgs.identity import IdentityClassifier, composite_identity_along_ray, composite_identity_batched
from ltx_trainer.r5dgs.lookup import ObjectEmbeddingLookup
from ltx_trainer.r5dgs.losses import (
    loss_3d_neighbor_kl,
    loss_majority_consistency,
    loss_obj_2d,
    loss_rigid_distance,
    render_loss_l1_ssim,
    total_loss_r5dgs,
)
from ltx_trainer.r5dgs.quaternion import quat_mul, quat_normalize, quat_rotate_vector, quat_to_rotmat
from ltx_trainer.r5dgs.pipeline import (
    benchmark_manifest,
    evaluation_demo,
    framework_card,
    method_variants,
    paper_limitations,
    table_fps_miou,
    table_reconstruction_metrics,
    training_step_demo,
)
from ltx_trainer.r5dgs.rigid import (
    canonical_offsets,
    knn_indices_bruteforce,
    propagate_rigid_positions,
    propagate_rigid_quaternions,
    representative_indices,
)

__all__ = [
    "R5DGSConfig",
    "IdentityClassifier",
    "ObjectEmbeddingLookup",
    "canonical_offsets",
    "composite_identity_along_ray",
    "composite_identity_batched",
    "knn_indices_bruteforce",
    "loss_3d_neighbor_kl",
    "loss_majority_consistency",
    "loss_obj_2d",
    "loss_rigid_distance",
    "propagate_rigid_positions",
    "propagate_rigid_quaternions",
    "quat_mul",
    "quat_normalize",
    "quat_rotate_vector",
    "quat_to_rotmat",
    "render_loss_l1_ssim",
    "representative_indices",
    "total_loss_r5dgs",
    "benchmark_manifest",
    "evaluation_demo",
    "framework_card",
    "method_variants",
    "paper_limitations",
    "table_fps_miou",
    "table_reconstruction_metrics",
    "training_step_demo",
]
