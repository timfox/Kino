"""Synthetic training / streaming demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.evogs.compression import compress_tree_refinements
from ltx_trainer.evogs.config import EvoGSConfig, RefinementMode
from ltx_trainer.evogs.metrics import l1_dssim_loss, psnr, ssim_proxy
from ltx_trainer.evogs.progressive import progressive_train_demo
from ltx_trainer.evogs.refinement import refinement_summary, split_children
from ltx_trainer.evogs.streaming import streaming_demo
from ltx_trainer.evogs.tree import EvolutionTree, synthetic_tree


def refinement_option_demo(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dim = 59
    parent = rng.normal(scale=0.1, size=dim)
    psi = rng.normal(scale=0.02, size=dim)
    alpha = rng.uniform(0.5, 1.5, size=5)
    out: dict[str, Any] = {}
    for mode in RefinementMode:
        c1, c2 = split_children(parent, psi, alpha, mode=mode)
        out[mode.value] = {
            "child_delta_l2": float(np.linalg.norm(c1 - c2)),
            "psi_summary": refinement_summary(psi),
        }
    return out


def render_loss_stub(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    gt = rng.random((32, 32))
    coarse = gt + rng.normal(scale=0.06, size=gt.shape)
    fine_parent = gt + rng.normal(scale=0.04, size=gt.shape)
    dim = 1
    psi = rng.normal(scale=0.02, size=())
    c1, c2 = split_children(np.array([fine_parent.mean()]), np.array([psi]), np.array([1.0]))
    refined = 0.5 * (c1 + c2)
    return {
        "L0_psnr": psnr(gt, coarse),
        "L0_ssim": ssim_proxy(gt, coarse),
        "L1_psnr": psnr(gt, fine_parent),
        "refined_psnr": psnr(gt, np.full_like(gt, float(refined[0]))),
        "loss_L0": l1_dssim_loss(gt, coarse),
        "loss_L1": l1_dssim_loss(gt, fine_parent),
    }


def full_pipeline_demo(cfg: EvoGSConfig | None = None, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or EvoGSConfig()
    tree = synthetic_tree(n_roots=40, splits_per_level=10, levels=4, cfg=cfg, seed=seed)
    psis = []
    for n in tree.nodes.values():
        if n.parent_id is not None and np.any(n.psi):
            sparse = n.psi.copy()
            sparse[np.abs(sparse) < np.percentile(np.abs(sparse), 80)] = 0.0
            psis.append(sparse)
    comp = compress_tree_refinements(psis[: min(32, len(psis))])
    prog = progressive_train_demo(cfg=cfg, seed=seed)
    stream = streaming_demo(seed=seed)
    return {
        "tree": tree.to_dict(),
        "refinement_options": refinement_option_demo(seed),
        "render_loss": render_loss_stub(seed),
        "progressive": prog,
        "compression": comp,
        "streaming": stream,
        "ghost_ratio": tree.ghost_ratio(),
    }


def train_step_torch(seed: int = 0) -> dict[str, Any]:
    """Optional torch backward on refinement ψ (smoke)."""
    try:
        import torch
    except ImportError:
        return {"torch": False, "skipped": True}
    torch.manual_seed(seed)
    parent = torch.randn(59, requires_grad=True)
    psi = (torch.randn(59) * 0.02).requires_grad_(True)
    alpha = torch.tensor([1.0])
    c1 = parent + psi
    c2 = parent - alpha[0] * psi
    target = torch.randn(59)
    loss = ((c1 - target) ** 2).mean() + ((c2 - target) ** 2).mean()
    loss.backward()
    return {
        "torch": True,
        "loss": float(loss.detach()),
        "psi_grad_norm": float(psi.grad.norm()) if psi.grad is not None else 0.0,
    }
