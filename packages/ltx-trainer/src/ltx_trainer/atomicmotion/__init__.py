"""AtomicMotion: sparse HMD to full-body pose (Liu et al., arXiv:2605.22631)."""

from ltx_trainer.atomicmotion.config import BASELINES, PARTITION_NAMES, AtomicMotionConfig
from ltx_trainer.atomicmotion.gsm import GlobalSynchronizedModulation
from ltx_trainer.atomicmotion.kinematic import KinematicAttention, TKBlock, star_topology_weights
from ltx_trainer.atomicmotion.losses import pose_losses, shape_consistency_loss
from ltx_trainer.atomicmotion.metrics import evaluate_pose_sequence, mpjpe, mpjre, mpjve
from ltx_trainer.atomicmotion.mpmg import (
    curriculum_mask_ratio,
    masked_pose_input,
    sparse_tracking_input,
)
from ltx_trainer.atomicmotion.partitions import (
    OBSERVABLE_JOINTS,
    atomic_intent_partition,
    merge_partitions,
)
from ltx_trainer.atomicmotion.pipeline import (
    AtomicMotionModel,
    ablation_table,
    benchmark_table_amass_p1,
    dataset_card,
    demo_motion_batch,
    scalability_table,
    training_step,
)

__all__ = [
    "BASELINES",
    "AtomicMotionConfig",
    "AtomicMotionModel",
    "GlobalSynchronizedModulation",
    "KinematicAttention",
    "OBSERVABLE_JOINTS",
    "PARTITION_NAMES",
    "TKBlock",
    "ablation_table",
    "atomic_intent_partition",
    "benchmark_table_amass_p1",
    "curriculum_mask_ratio",
    "dataset_card",
    "demo_motion_batch",
    "evaluate_pose_sequence",
    "masked_pose_input",
    "merge_partitions",
    "mpjpe",
    "mpjre",
    "mpjve",
    "pose_losses",
    "scalability_table",
    "shape_consistency_loss",
    "sparse_tracking_input",
    "star_topology_weights",
    "training_step",
]
