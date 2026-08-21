"""R5DGS framework card, paper tables (II–III), and smoke demos (Gridusov et al., arXiv:2605.25909)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.r5dgs.config import R5DGSConfig
from ltx_trainer.r5dgs.identity import IdentityClassifier, composite_identity_batched
from ltx_trainer.r5dgs.lookup import ObjectEmbeddingLookup
from ltx_trainer.r5dgs.losses import (
    loss_3d_neighbor_kl,
    loss_majority_consistency,
    loss_obj_2d,
    total_loss_r5dgs,
)
from ltx_trainer.r5dgs.rigid import (
    canonical_offsets,
    knn_indices_bruteforce,
    propagate_rigid_positions,
    representative_indices,
)


def framework_card(cfg: R5DGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or R5DGSConfig()
    return {
        "name": "R5DGS",
        "paper": "arXiv:2605.25909",
        "title": "Semantic-Aware 4D Gaussian Splatting with Rigid Body Constraints",
        "authors": "Gridusov, Popov, Kolyubin (ITMO BE2R)",
        "builds_on": "TRACE (Li, Song & Yang, ICCV 2025)",
        "identity_dim": cfg.identity_dim,
        "losses": "L5DGS = L_render + λ_obj L_obj + λ_3d L_3d (+ L_rigid, L_major gated)",
        "inference": "Rigid-body: O(K) TRD reps vs O(N) per-Gaussian (Table I variants)",
        "open_vocab": "Offline CLIP / Perception Encoder lookup (Eq. 8)",
        "mask_pipeline": "SAM2 + DEVA (external)",
        "benchmark": "Dynamic Indoor Scene dataset (NVFI)",
        "reported_speedup_fps": 11.0,
    }


def method_variants() -> list[dict[str, str]]:
    """Table I — training losses vs inference strategy."""
    return [
        {
            "variant": "5DGS",
            "training": "L5DGS",
            "inference": "Standard (per-Gaussian TRD)",
        },
        {
            "variant": "R5DGS",
            "training": "L5DGS",
            "inference": "Rigid-body (representative + propagation)",
        },
        {
            "variant": "R5DGS w/ extra loss",
            "training": "L5DGS + L_rigid + L_major",
            "inference": "Rigid-body",
        },
    ]


def table_reconstruction_metrics() -> dict[str, dict[str, dict[str, float]]]:
    """Table II — PSNR / SSIM / LPIPS on four Dynamic Indoor scenes."""
    scenes = ("Dining Table", "Chessboard", "Darkroom", "Factory")
    methods = {
        "TRACE": {
            "Dining Table": (35.580, 0.962, 0.0497),
            "Chessboard": (34.630, 0.963, 0.055),
            "Darkroom": (37.774, 0.961, 0.067),
            "Factory": (36.488, 0.965, 0.049),
        },
        "5DGS": {
            "Dining Table": (35.428, 0.956, 0.055),
            "Chessboard": (33.991, 0.956, 0.063),
            "Darkroom": (36.600, 0.955, 0.074),
            "Factory": (35.926, 0.958, 0.055),
        },
        "R5DGS": {
            "Dining Table": (28.844, 0.942, 0.066),
            "Chessboard": (28.805, 0.932, 0.086),
            "Darkroom": (31.181, 0.939, 0.091),
            "Factory": (29.798, 0.924, 0.075),
        },
        "R5DGS w/ extra loss": {
            "Dining Table": (28.688, 0.939, 0.067),
            "Chessboard": (29.153, 0.929, 0.089),
            "Darkroom": (31.537, 0.943, 0.087),
            "Factory": (30.749, 0.928, 0.073),
        },
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for method, per_scene in methods.items():
        out[method] = {}
        for scene in scenes:
            psnr, ssim, lpips = per_scene[scene]
            out[method][scene] = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    return out


def table_fps_miou() -> dict[str, dict[str, float | dict[str, float]]]:
    """Table III — extrapolation FPS and mIoU (5DGS vs R5DGS)."""
    per_scene = {
        "5DGS": {
            "Dining Table": {"fps": 66.9, "miou": 0.78},
            "Chessboard": {"fps": 67.3, "miou": 0.75},
            "Darkroom": {"fps": 49.4, "miou": 0.37},
            "Factory": {"fps": 64.9, "miou": 0.47},
            "Overall": {"fps": 62.1, "miou": 0.59},
        },
        "R5DGS": {
            "Dining Table": {"fps": 76.3, "miou": 0.77},
            "Chessboard": {"fps": 76.9, "miou": 0.73},
            "Darkroom": {"fps": 66.2, "miou": 0.38},
            "Factory": {"fps": 75.0, "miou": 0.46},
            "Overall": {"fps": 73.6, "miou": 0.59},
        },
    }
    five = per_scene["5DGS"]["Overall"]["fps"]
    r5 = per_scene["R5DGS"]["Overall"]["fps"]
    return {
        **per_scene,
        "fps_delta_overall": r5 - five,
        "representative_groups_typical": 9.0,
        "gaussians_typical": 40_000.0,
    }


def paper_limitations() -> list[str]:
    return [
        "Rigid-body inference trades PSNR for ~11 FPS extrapolation speedup vs per-Gaussian TRD.",
        "Requires offline SAM2+DEVA masks; tracking errors hurt Darkroom/Factory mIoU.",
        "TRACE TRD integrator and differentiable 3DGS rasterizer are external dependencies.",
        "Highly deformable regions violate rigid propagation assumptions.",
    ]


def benchmark_manifest(cfg: R5DGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or R5DGSConfig()
    return {
        "dataset": "Dynamic Indoor Scene (NVFI)",
        "scenes": ["Dining Table", "Chessboard", "Darkroom", "Factory"],
        "metrics": ["PSNR", "SSIM", "LPIPS", "mIoU", "FPS"],
        "variants": [v["variant"] for v in method_variants()],
        "identity_dim": cfg.identity_dim,
        "mask_pipeline": "SAM2 + DEVA",
    }


def training_step_demo(cfg: R5DGSConfig | None = None) -> dict[str, float]:
    """Smoke: identity composite, k-NN regularizers, and gated total loss."""
    cfg = cfg or R5DGSConfig()
    torch.manual_seed(7)
    n = 32
    ident = torch.randn(n, cfg.identity_dim)
    alpha = torch.sigmoid(torch.randn(n))
    e_ray = composite_identity_batched(ident.unsqueeze(0), alpha.unsqueeze(0))[0]
    clf = IdentityClassifier(cfg)
    logits = clf(e_ray.unsqueeze(0))
    labels = torch.zeros(1, dtype=torch.long)
    l_obj = loss_obj_2d(logits, labels)

    xyz = torch.randn(n, 3)
    nbr = knn_indices_bruteforce(xyz, k=cfg.k_neighbors)
    l3d = loss_3d_neighbor_kl(ident, nbr)
    logits_all = clf(ident)
    lmaj = loss_majority_consistency(logits_all, nbr)

    tot_early = total_loss_r5dgs(
        torch.tensor(0.05),
        l_obj,
        l3d,
        lambda_obj=cfg.lambda_obj,
        lambda_3d=cfg.lambda_3d,
        iteration=0,
        tau_reg=cfg.tau_reg,
        l_major=lmaj,
        lambda_maj=cfg.lambda_maj,
    )
    tot_late = total_loss_r5dgs(
        torch.tensor(0.05),
        l_obj,
        l3d,
        lambda_obj=cfg.lambda_obj,
        lambda_3d=cfg.lambda_3d,
        iteration=cfg.t_rigid + 1,
        tau_reg=cfg.tau_reg,
        l_rigid=torch.tensor(0.01),
        lambda_rigid=cfg.lambda_rigid,
        l_major=lmaj,
        lambda_maj=cfg.lambda_maj,
        t_rigid=cfg.t_rigid,
    )
    return {
        "loss_obj": l_obj.detach().item(),
        "loss_3d": l3d.detach().item(),
        "loss_major": lmaj.detach().item(),
        "total_early": tot_early.detach().item(),
        "total_after_t_rigid": tot_late.detach().item(),
        "identity_dim": float(cfg.identity_dim),
        "k_neighbors": float(cfg.k_neighbors),
    }


def evaluation_demo(cfg: R5DGSConfig | None = None) -> dict[str, Any]:
    """Smoke: rigid propagation, lookup retrieval, and paper benchmark deltas."""
    cfg = cfg or R5DGSConfig()
    step = training_step_demo(cfg)

    x_def = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gid = torch.zeros(3, dtype=torch.long)
    rep = representative_indices(x_def, gid, num_groups=1)
    offsets = canonical_offsets(x_def, rep)
    x_vel_at = x_def[rep] + torch.tensor([0.3, 0.0, 0.0])
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    x_new = propagate_rigid_positions(x_def, x_vel_at, q_id, offsets, rep)

    lut = ObjectEmbeddingLookup(num_groups=4, embed_dim=cfg.lookup_embed_dim)
    with torch.no_grad():
        lut.table[2] = 1.0
    retrieved = lut.retrieve_group(lut.table[2].clone())

    fps_tab = table_fps_miou()
    recon = table_reconstruction_metrics()

    return {
        **step,
        "rigid_propagated_finite": bool(torch.isfinite(x_new).all()),
        "lookup_group": retrieved,
        "fps_5dgs_overall": fps_tab["5DGS"]["Overall"]["fps"],
        "fps_r5dgs_overall": fps_tab["R5DGS"]["Overall"]["fps"],
        "fps_delta_overall": fps_tab["fps_delta_overall"],
        "miou_preserved": fps_tab["5DGS"]["Overall"]["miou"] == fps_tab["R5DGS"]["Overall"]["miou"],
        "r5dgs_psnr_dining": recon["R5DGS"]["Dining Table"]["psnr"],
        "trace_psnr_dining": recon["TRACE"]["Dining Table"]["psnr"],
        "variants": method_variants(),
    }
