"""Configuration for AtomicMotion (Liu et al., arXiv:2605.22631)."""

from __future__ import annotations

from dataclasses import dataclass


PARTITION_NAMES: tuple[str, ...] = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")
BASELINES: tuple[str, ...] = ("sage", "egoposer", "hmd_poser", "rpm")


@dataclass
class AtomicMotionConfig:
    """Defaults from paper Sec. 3–4 and Appendix A."""

    num_joints: int = 22
    feature_dim: int = 18  # pos(3)+vel(3)+rot6d(6)+ang_vel6d(6)
    embed_dim: int = 256
    num_tk_blocks: int = 6
    num_heads: int = 8
    shape_dim: int = 16
    rot_dim: int = 6
    window_40: int = 40
    window_80: int = 80
    mpmg_mask_prob: float = 0.5
    mpmg_initial_masked_ratio: float = 0.8
    mpmg_decay_steps: int = 50_000
    # Table 1 AMASS-P1 reference (AtomicMotion-80)
    reference_mpjre_p1: float = 2.27
    reference_mpjpe_p1: float = 3.04
    reference_mpjve_p1: float = 14.53
    reference_jitter_p1: float = 5.68
