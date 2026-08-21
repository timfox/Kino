"""AtomicMotion pipeline glue (Sec. 3, Fig. 2)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.atomicmotion.config import BASELINES, AtomicMotionConfig
from ltx_trainer.atomicmotion.gsm import GlobalSynchronizedModulation
from ltx_trainer.atomicmotion.kinematic import TKBlock
from ltx_trainer.atomicmotion.losses import pose_losses, shape_consistency_loss
from ltx_trainer.atomicmotion.metrics import evaluate_pose_sequence
from ltx_trainer.atomicmotion.mpmg import masked_pose_input, sparse_tracking_input
from ltx_trainer.atomicmotion.partitions import (
    PARTITION_JOINTS,
    PARTITION_NAMES,
    PartitionName,
    atomic_intent_partition,
    merge_partitions,
)


class DecoupledAIPDecoder(nn.Module):
    """Dedicated MLP heads per partition with limb weight sharing (Eq. 10)."""

    def __init__(self, cfg: AtomicMotionConfig | None = None):
        super().__init__()
        cfg = cfg or AtomicMotionConfig()
        self.torso = nn.Linear(cfg.embed_dim, len(PARTITION_JOINTS["torso"]) * cfg.rot_dim)
        self.limb = nn.Linear(cfg.embed_dim, len(PARTITION_JOINTS["left_arm"]) * cfg.rot_dim)

    def forward(self, parts: dict[PartitionName, Tensor]) -> Tensor:
        chunks: list[tuple[tuple[int, ...], Tensor]] = []
        for name in PARTITION_NAMES:
            feat = parts[name]
            if name == "torso":
                rot = self.torso(feat)
            else:
                rot = self.limb(feat)
            n_j = len(PARTITION_JOINTS[name])
            if feat.dim() == 2:
                rot = rot.view(-1, n_j, 6)
            else:
                rot = rot.view(feat.shape[0], -1, n_j, 6)
            chunks.append((PARTITION_JOINTS[name], rot))
        # merge
        if parts["torso"].dim() == 2:
            t, _ = parts["torso"].shape
            out = torch.zeros(t, 22, 6, device=parts["torso"].device)
            for joints, rot in chunks:
                out[:, joints, :] = rot
            return out
        b, t = parts["torso"].shape[:2]
        out = torch.zeros(b, t, 22, 6, device=parts["torso"].device)
        for joints, rot in chunks:
            out[:, :, joints, :] = rot
        return out


class AtomicMotionModel(nn.Module):
    """Simplified end-to-end stack for smoke tests."""

    def __init__(self, cfg: AtomicMotionConfig | None = None):
        super().__init__()
        cfg = cfg or AtomicMotionConfig()
        self.cfg = cfg
        self.gsm = GlobalSynchronizedModulation(cfg)
        self.blocks = nn.ModuleList([TKBlock(cfg) for _ in range(cfg.num_tk_blocks)])
        self.decoder = DecoupledAIPDecoder(cfg)
        self.shape_head = nn.Linear(cfg.embed_dim * 5, cfg.shape_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        parts = self.gsm(x)
        for block in self.blocks:
            parts = block(parts)
        pose = self.decoder(parts)
        stacked = torch.cat([parts[n] for n in PARTITION_NAMES], dim=-1)
        if stacked.dim() == 2:
            beta = self.shape_head(stacked.mean(dim=0, keepdim=True))
        else:
            beta = self.shape_head(stacked.mean(dim=1))
        return pose, beta


def demo_motion_batch(
    *,
    frames: int = 40,
    batch: int = 2,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    device = device or torch.device("cpu")
    x_sparse = torch.randn(batch, frames, 22, 18, device=device)
    x_full = torch.randn(batch, frames, 22, 18, device=device)
    return x_sparse, x_full


def training_step(
    model: AtomicMotionModel,
    x_full: Tensor,
    *,
    step: int = 0,
    cfg: AtomicMotionConfig | None = None,
) -> dict[str, float]:
    cfg = cfg or AtomicMotionConfig()
    from ltx_trainer.atomicmotion.mpmg import curriculum_mask_ratio

    ratio = curriculum_mask_ratio(step, initial=cfg.mpmg_initial_masked_ratio, decay_steps=cfg.mpmg_decay_steps)
    if ratio > 0 and torch.rand(1).item() < ratio:
        x_in = masked_pose_input(x_full, mask_prob=cfg.mpmg_mask_prob)
    else:
        x_in = sparse_tracking_input(x_full)
    pred_pose, pred_beta = model(x_in)
    gt_pose = x_full[..., -6:]  # demo: last 6 dims as rot proxy
    losses = pose_losses(pred_pose, gt_pose, pred_pose[..., :3], gt_pose[..., :3])
    losses["l_consist"] = shape_consistency_loss(pred_beta)
    losses["loss"] = losses["loss"] + losses["l_consist"]
    return {k: float(v.item()) if isinstance(v, Tensor) else v for k, v in losses.items()}


def benchmark_table_amass_p1() -> dict[str, dict[str, float]]:
    """Table 1 AMASS-P1."""
    return {
        "sage": {"mpjre": 2.49, "mpjpe": 3.18, "mpjve": 20.78, "jitter": 6.46},
        "egoposer": {"mpjre": 2.90, "mpjpe": 3.80, "mpjve": 24.26, "jitter": 14.89},
        "hmd_poser": {"mpjre": 2.29, "mpjpe": 3.15, "mpjve": 17.47, "jitter": 6.43},
        "rpm": {"mpjre": 3.25, "mpjpe": 4.08, "mpjve": 19.21, "jitter": 4.21},
        "atomicmotion_40": {"mpjre": 2.20, "mpjpe": 3.00, "mpjve": 16.08, "jitter": 6.63},
        "atomicmotion_80": {"mpjre": 2.27, "mpjpe": 3.04, "mpjve": 14.53, "jitter": 5.68},
    }


def ablation_table() -> dict[str, dict[str, float]]:
    """Table 3 AMASS-P1 ablations."""
    return {
        "wo_mpmg": {"mpjre": 2.307, "mpjpe": 3.103, "mpjve": 14.745, "jitter": 5.687},
        "wo_gsm": {"mpjre": 2.283, "mpjpe": 3.051, "mpjve": 14.401, "jitter": 5.520},
        "wo_intent_branch": {"mpjre": 2.292, "mpjpe": 3.083, "mpjve": 14.593, "jitter": 5.492},
        "wo_structural_branch": {"mpjre": 2.267, "mpjpe": 3.052, "mpjve": 14.711, "jitter": 5.834},
        "wo_decoupled_decoder": {"mpjre": 2.259, "mpjpe": 3.036, "mpjve": 15.056, "jitter": 6.055},
        "ours": {"mpjre": 2.265, "mpjpe": 3.040, "mpjve": 14.531, "jitter": 5.676},
    }


def scalability_table() -> dict[str, dict[str, float]]:
    """Table 4 sensor scalability."""
    return {
        "hmd": {"mpjre": 2.27, "mpjpe": 3.04, "mpjve": 14.53, "jitter": 5.68},
        "hmd_1": {"mpjre": 1.82, "mpjpe": 1.83, "mpjve": 9.57, "jitter": 5.04},
        "hmd_2": {"mpjre": 1.55, "mpjpe": 1.18, "mpjve": 5.86, "jitter": 3.63},
        "hmd_3": {"mpjre": 1.24, "mpjpe": 0.96, "mpjve": 4.03, "jitter": 3.54},
    }


def dataset_card(cfg: AtomicMotionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AtomicMotionConfig()
    return {
        "name": "AtomicMotion",
        "paper": "arXiv:2605.22631",
        "task": "full-body pose from sparse HMD + hand tracking",
        "dataset": "AMASS",
        "joints": cfg.num_joints,
        "partitions": list(PARTITION_NAMES),
        "feature_dim": cfg.feature_dim,
        "tk_blocks": cfg.num_tk_blocks,
        "baselines": list(BASELINES),
    }
