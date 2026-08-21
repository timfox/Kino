"""Full-4D framework card, paper tables, and smoke demos (arXiv:2605.25500)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.full4d.attention import masked_attention, tv_fused_mask
from ltx_trainer.full4d.config import Full4DConfig
from ltx_trainer.full4d.flow import cfm_loss, rectified_interpolate
from ltx_trainer.full4d.fmd import clean_estimate, corrupt_latent, fmd_loss
from ltx_trainer.full4d.gaussians import DeformationField, frame_dim_concat


def framework_card(cfg: Full4DConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Full4DConfig()
    return {
        "name": "Full-4D",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "dataset": "Real-MV-4D (6 synchronized cameras, 2k+ scenes)",
        "backbone": "Wan2.2-TI2V-5B + DA3 projection + 4D-GS",
        "mechanisms": [
            "T×V fused sparse attention (Eq. 10)",
            "Projection + camera conditioning",
            "FMD-regularized 4D Gaussian lifting",
        ],
    }


def table_mv_generation() -> dict[str, dict[str, float]]:
    """Table 1 — multi-view video generation metrics."""
    return {
        "Free4D": {"fid": 115.74, "fvd": 741.88, "clip_f": 96.74, "mat_pix_k": 109.01, "fvd_v": 680.65, "clip_v": 90.63},
        "DimensionX": {"fid": 98.69, "fvd": 577.84, "clip_f": 92.72, "mat_pix_k": 365.53, "fvd_v": 560.87, "clip_v": 91.37},
        "ReCamMaster": {"fid": 62.38, "fvd": 360.86, "clip_f": 98.36, "mat_pix_k": 448.72, "fvd_v": 383.28, "clip_v": 90.50},
        "Full-4D": {"fid": 35.12, "fvd": 175.29, "clip_f": 99.74, "mat_pix_k": 540.92, "fvd_v": 140.60, "clip_v": 93.17},
    }


def table_vbench() -> dict[str, dict[str, float]]:
    """Table 2 — VBench scores."""
    return {
        "Free4D": {"aesthetic": 41.34, "background": 90.21, "imaging": 61.43, "motion": 96.36, "subject": 89.00, "temporal": 95.83},
        "ReCamMaster": {"aesthetic": 48.99, "background": 88.44, "imaging": 65.15, "motion": 99.13, "subject": 87.11, "temporal": 97.92},
        "Full-4D": {"aesthetic": 46.26, "background": 96.13, "imaging": 70.80, "motion": 99.64, "subject": 97.77, "temporal": 99.63},
    }


def table_reconstruction() -> dict[str, dict[str, float]]:
    """Table 3 — Real-MV-4D and DyCheck PSNR/SSIM/LPIPS."""
    return {
        "Free4D": {"psnr_mv": 13.16, "ssim_mv": 0.436, "lpips_mv": 0.656, "psnr_dy": 11.83},
        "ReCamMaster": {"psnr_mv": 13.45, "ssim_mv": 0.396, "lpips_mv": 0.613, "psnr_dy": 12.44},
        "TrajectoryCrafter": {"psnr_mv": 13.60, "ssim_mv": 0.432, "lpips_mv": 0.592, "psnr_dy": 14.34},
        "Full-4D": {"psnr_mv": 15.50, "ssim_mv": 0.486, "lpips_mv": 0.654, "psnr_dy": 14.59},
    }


def table_ablation_generation() -> dict[str, dict[str, float]]:
    """Table 4 — multi-view generation ablations."""
    return {
        "w/o Fused Attention": {"mat_pix_k": 101.35, "fvd_v": 528.47, "clip_v": 90.84},
        "w/o Projection": {"mat_pix_k": 468.72, "fvd_v": 252.63, "clip_v": 92.26},
        "w/o Intra-time": {"mat_pix_k": 389.54, "fvd_v": 210.72, "clip_v": 91.02},
        "w/o Cross-half": {"mat_pix_k": 165.41, "fvd_v": 318.65, "clip_v": 89.47},
        "Full-4D": {"mat_pix_k": 540.92, "fvd_v": 140.60, "clip_v": 93.17},
    }


def table_ablation_fmd() -> dict[str, dict[str, float]]:
    """Table 5 — FMD ablation during 4DGS."""
    return {
        "w/o FMD": {"mat_pix_k": 256.28, "fvd_v": 768.57, "clip_v": 83.84},
        "Full-4D": {"mat_pix_k": 509.73, "fvd_v": 187.69, "clip_v": 90.93},
    }


def training_step_demo(cfg: Full4DConfig | None = None) -> dict[str, float]:
    """Smoke: CFM, TV mask attention, FMD, deformation."""
    cfg = cfg or Full4DConfig()
    torch.manual_seed(42)
    nv, f, d, s = 3, 4, cfg.token_dim, 8
    z0 = torch.randn(nv, f, s, d)
    eps = torch.randn_like(z0)
    t = torch.tensor([0.3])
    zt = rectified_interpolate(z0, eps, t)
    v_pred = eps - z0 + 0.01 * torch.randn_like(z0)
    loss_cfm = cfm_loss(v_pred, z0, eps)

    q = k = v = torch.randn(nv * 2 * f, d)
    mask = tv_fused_mask(nv, f)
    _ = masked_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask)

    z_proj = torch.randn(f, d)
    z_tgt = torch.randn(f, d)
    _ = frame_dim_concat(z_tgt, z_proj)

    z = torch.randn(2, d)
    tau = torch.tensor([0.5])
    z_tau = corrupt_latent(z, tau)
    z_hat = clean_estimate(z_tau, tau, torch.zeros_like(z))
    loss_fmd = fmd_loss(z, z_hat)

    deform = DeformationField()
    mu = torch.randn(5, 3)
    t_frame = torch.tensor(0.5)
    d_mu, _, _ = deform(mu, t_frame)

    return {
        "cfm_loss": float(loss_cfm.detach()),
        "mask_density": float(mask.mean()),
        "fmd_loss": float(loss_fmd.detach()),
        "deform_mu_norm": float(d_mu.detach().norm()),
    }


def evaluation_demo(cfg: Full4DConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering."""
    cfg = cfg or Full4DConfig()
    step = training_step_demo(cfg)
    t1 = table_mv_generation()
    t3 = table_reconstruction()
    t4 = table_ablation_generation()
    t5 = table_ablation_fmd()

    return {
        **step,
        "best_fid": t1["Full-4D"]["fid"] == min(m["fid"] for m in t1.values()),
        "best_fvd": t1["Full-4D"]["fvd"] == min(m["fvd"] for m in t1.values()),
        "best_mat_pix": t1["Full-4D"]["mat_pix_k"] == max(m["mat_pix_k"] for m in t1.values()),
        "best_psnr_mv": t3["Full-4D"]["psnr_mv"] == max(m["psnr_mv"] for m in t3.values()),
        "fused_attn_critical": t4["Full-4D"]["mat_pix_k"] > t4["w/o Fused Attention"]["mat_pix_k"],
        "fmd_improves_sync": t5["Full-4D"]["mat_pix_k"] > t5["w/o FMD"]["mat_pix_k"],
        "six_camera_setup": float(cfg.num_views) == 6.0,
    }
