"""DRFusion framework card, paper tables, and smoke demos (arXiv:2605.25775)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.drfusion.config import DRFusionConfig
from ltx_trainer.drfusion.history import stabilized_history_guidance
from ltx_trainer.drfusion.losses import fusion_pixel_loss, latent_temporal_loss
from ltx_trainer.drfusion.model import DRFusionVelocityHead


def framework_card(cfg: DRFusionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DRFusionConfig()
    return {
        "name": "DRFusion",
        "paper": cfg.paper_arxiv,
        "venue": "ICML 2026 (PMLR 306)",
        "code": cfg.code_url,
        "paradigm": "history-conditioned 3D-DiT motion generation",
        "mechanisms": [
            "Stabilized History Guidance",
            "Soft Temporal Anchoring",
            "Decoupled Structure-Motion Adaptation",
        ],
        "training": f"Stage I VAE ({cfg.stage1_epochs} ep) + Stage II adapter ({cfg.stage2_epochs} ep)",
        "inference": f"DDIM {cfg.ddim_steps} steps, history T={cfg.history_window}, s={cfg.guidance_scale}",
        "history_window": cfg.history_window,
        "guidance_scale": cfg.guidance_scale,
        "datasets": list(cfg.datasets),
    }


def table_fusion_quality() -> dict[str, dict[str, dict[str, float]]]:
    """Table 1 — CC, EN, SSIM (↑)."""
    return {
        "HDO": {
            "DDFM": {"cc": 0.679, "en": 6.196, "ssim": 0.585},
            "UniVF": {"cc": 0.632, "en": 6.313, "ssim": 0.573},
            "TemCOCO": {"cc": 0.585, "en": 6.247, "ssim": 0.580},
            "Ours": {"cc": 0.682, "en": 6.692, "ssim": 0.595},
        },
        "M3SVD": {
            "DDFM": {"cc": 0.604, "en": 6.439, "ssim": 0.629},
            "UniVF": {"cc": 0.560, "en": 6.525, "ssim": 0.629},
            "Ours": {"cc": 0.594, "en": 6.574, "ssim": 0.635},
        },
        "NOT-156": {
            "DDFM": {"cc": 0.441, "en": 6.637, "ssim": 0.586},
            "TemCOCO": {"cc": 0.236, "en": 6.558, "ssim": 0.397},
            "Ours": {"cc": 0.458, "en": 6.664, "ssim": 0.612},
        },
        "VTMOT": {
            "DDFM": {"cc": 0.623, "en": 6.241, "ssim": 0.440},
            "UniVF": {"cc": 0.590, "en": 6.611, "ssim": 0.440},
            "Ours": {"cc": 0.639, "en": 6.783, "ssim": 0.450},
        },
    }


def table_temporal_stability() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — BiSWE, MS2R (↓)."""
    return {
        "HDO": {
            "TemCOCO": {"biswe": 6.414, "ms2r": 0.213},
            "UniVF": {"biswe": 6.378, "ms2r": 0.227},
            "Ours": {"biswe": 6.225, "ms2r": 0.211},
        },
        "M3SVD": {
            "TemCOCO": {"biswe": 6.488, "ms2r": 0.258},
            "UniVF": {"biswe": 7.316, "ms2r": 0.222},
            "Ours": {"biswe": 6.467, "ms2r": 0.220},
        },
        "NOT-156": {
            "TemCOCO": {"biswe": 4.593, "ms2r": 0.382},
            "UniVF": {"biswe": 4.690, "ms2r": 0.338},
            "Ours": {"biswe": 4.816, "ms2r": 0.332},
        },
        "VTMOT": {
            "TemCOCO": {"biswe": 8.122, "ms2r": 0.826},
            "UniVF": {"biswe": 8.551, "ms2r": 0.591},
            "Ours": {"biswe": 7.418, "ms2r": 0.560},
        },
    }


def table_ablation_not156() -> dict[str, dict[str, float]]:
    """Table 3 — NOT-156 ablation."""
    return {
        "w/o L_temp": {"cc": 0.442, "en": 6.523, "ssim": 0.589, "biswe": 5.172, "ms2r": 0.358},
        "w/o Adapter": {"cc": 0.432, "en": 6.587, "ssim": 0.544, "biswe": 5.214, "ms2r": 0.342},
        "w/o LR.": {"cc": 0.328, "en": 5.932, "ssim": 0.547, "biswe": 4.911, "ms2r": 0.342},
        "w/o HG.": {"cc": 0.438, "en": 6.487, "ssim": 0.571, "biswe": 5.238, "ms2r": 0.361},
        "w/o H(2)": {"cc": 0.440, "en": 6.501, "ssim": 0.601, "biswe": 4.972, "ms2r": 0.341},
        "Ours": {"cc": 0.458, "en": 6.664, "ssim": 0.612, "biswe": 4.816, "ms2r": 0.332},
    }


def table_object_tracking() -> dict[str, dict[str, float]]:
    """Table 4 — ByteTrack on NOT-156 fused videos."""
    return {
        "DDFM": {"auc": 0.240, "dp20": 0.215, "sr05": 0.221, "sr075": 0.090},
        "UniVF": {"auc": 0.212, "dp20": 0.217, "sr05": 0.196, "sr075": 0.071},
        "TemCOCO": {"auc": 0.185, "dp20": 0.187, "sr05": 0.150, "sr075": 0.077},
        "Ours": {"auc": 0.252, "dp20": 0.230, "sr05": 0.263, "sr075": 0.121},
    }


def training_step_demo(cfg: DRFusionConfig | None = None) -> dict[str, float]:
    """Smoke: L_temp + adapter fusion + history-guided velocity."""
    cfg = cfg or DRFusionConfig()
    torch.manual_seed(6)
    b, c, h, w, t = 2, 8, 32, 32, 4
    z_prev = torch.randn(b, c, h, w)
    z_curr = z_prev + 0.05 * torch.randn(b, c, h, w)
    flow = torch.randn(b, 2, h, w) * 0.1
    lt = latent_temporal_loss(z_prev, z_curr, flow)

    ir = torch.rand(b, 1, h, w)
    vis = torch.rand(b, 1, h, w)
    pred = 0.5 * ir + 0.5 * vis + 0.02 * torch.randn_like(vis)
    lf, parts = fusion_pixel_loss(pred, ir, vis, cfg)

    z_vis = torch.randn(b, c, t, h, w)
    hist = torch.randn(b, c, t, h, w)
    model = DRFusionVelocityHead(channels=c, cfg=cfg)
    v_guided, vels = model(z_vis, ir, hist)

    v0, v1, v2 = vels["H0"][:, :, -1], vels["H1"][:, :, -1], vels["H2"][:, :, -1]
    v_check = stabilized_history_guidance(v0, v1, v2, cfg.guidance_scale)

    return {
        "l_temp": float(lt.detach()),
        "fusion_loss": float(lf.detach()),
        **parts,
        "v_guided_mse_vs_formula": float(F.mse_loss(v_guided, v_check).detach()),
        "history_window": float(cfg.history_window),
    }


def evaluation_demo(cfg: DRFusionConfig | None = None) -> dict[str, Any]:
    """Smoke: verify paper table ordering."""
    cfg = cfg or DRFusionConfig()
    step = training_step_demo(cfg)
    fq = table_fusion_quality()
    ts = table_temporal_stability()
    abl = table_ablation_not156()
    trk = table_object_tracking()

    ssim_all_first = all(
        fq[d]["Ours"]["ssim"] >= max(fq[d][m]["ssim"] for m in fq[d] if m != "Ours")
        for d in fq
    )

    return {
        **step,
        "ssim_first_all_datasets": ssim_all_first,
        "ours_best_biswe_hdo": ts["HDO"]["Ours"]["biswe"] <= ts["HDO"]["UniVF"]["biswe"],
        "ours_best_ms2r_vtmot": ts["VTMOT"]["Ours"]["ms2r"] < ts["VTMOT"]["TemCOCO"]["ms2r"],
        "ablation_full_best_ssim": abl["Ours"]["ssim"] == max(r["ssim"] for r in abl.values()),
        "ablation_full_best_biswe": abl["Ours"]["biswe"] == min(r["biswe"] for r in abl.values()),
        "tracking_best_auc": trk["Ours"]["auc"] > max(trk[m]["auc"] for m in trk if m != "Ours"),
        "tracking_best_sr075": trk["Ours"]["sr075"] > trk["DDFM"]["sr075"],
        "v_guided_matches_eq6": step["v_guided_mse_vs_formula"] < 1e-6,
    }
