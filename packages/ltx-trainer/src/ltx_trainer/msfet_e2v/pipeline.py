"""MSFET-E2V framework card, paper tables, and smoke demos (arXiv:2605.25804)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.msfet_e2v.config import MSFETE2VConfig
from ltx_trainer.msfet_e2v.losses import total_loss
from ltx_trainer.msfet_e2v.metrics import psnr, ssim_proxy
from ltx_trainer.msfet_e2v.model import MSFETE2V
from ltx_trainer.msfet_e2v.voxel import events_to_voxel
from ltx_trainer.msfet_e2v.wavelet import haar_dwt2d, haar_idwt2d


def framework_card(cfg: MSFETE2VConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MSFETE2VConfig()
    return {
        "name": "MSFET-E2V",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "modules": ["CDAM (cross-domain attention + DWT)", "WSB (wavelet skip)", "RGD (residual decoder)"],
        "voxel_bins": cfg.voxel_bins,
        "embed_dim": cfg.embed_dim,
        "encoder_depth": cfg.encoder_depth,
        "parameters_m": cfg.num_parameters_m,
        "loss": "LPIPS + λ_TC temporal consistency (Eq. 7)",
        "training": f"{cfg.train_sequences} ESIM sequences, {cfg.train_epochs} epochs",
        "benchmarks": list(cfg.training_datasets),
    }


def table_quantitative_e2v() -> dict[str, dict[str, dict[str, float]]]:
    """Table I — ECD, HQF, MVSEC (PSNR↑ SSIM↑ LPIPS↓)."""
    return {
        "ECD": {
            "E2VID": {"psnr": 12.64, "ssim": 0.450, "lpips": 0.322},
            "FireNet": {"psnr": 12.33, "ssim": 0.479, "lpips": 0.320},
            "HyperE2VID": {"psnr": 14.10, "ssim": 0.494, "lpips": 0.326},
            "ET-Net": {"psnr": 13.52, "ssim": 0.521, "lpips": 0.275},
            "MSFET-E2V": {"psnr": 13.98, "ssim": 0.584, "lpips": 0.252},
        },
        "HQF": {
            "E2VID": {"psnr": 10.54, "ssim": 0.468, "lpips": 0.371},
            "ET-Net": {"psnr": 14.28, "ssim": 0.502, "lpips": 0.331},
            "MSFET-E2V": {"psnr": 14.82, "ssim": 0.548, "lpips": 0.291},
        },
        "MVSEC": {
            "E2VID": {"psnr": 6.98, "ssim": 0.241, "lpips": 0.644},
            "ET-Net": {"psnr": 10.15, "ssim": 0.311, "lpips": 0.568},
            "MSFET-E2V": {"psnr": 11.42, "ssim": 0.362, "lpips": 0.501},
        },
    }


def table_brisque() -> dict[str, dict[str, float]]:
    """Table II — no-reference BRISQUE (↓)."""
    return {
        "E2VID": {"mvsec_night": 19.16, "hdr": 22.83},
        "HyperE2VID": {"mvsec_night": 16.59, "hdr": 11.89},
        "ET-Net": {"mvsec_night": 16.17, "hdr": 18.47},
        "MSFET-E2V": {"mvsec_night": 9.32, "hdr": 7.40},
    }


def table_inference_efficiency() -> dict[str, dict[str, float]]:
    """Table III — parameters (M) and GPU ms per resolution."""
    return {
        "E2VID+": {"params_m": 10.71, "ms_240x180": 5.7, "ms_340x260": 10.1, "ms_640x480": 30.0},
        "HyperE2VID": {"params_m": 10.15, "ms_240x180": 6.6, "ms_340x260": 12.9, "ms_640x480": 22.0},
        "ET-Net": {"params_m": 22.18, "ms_240x180": 32.1, "ms_340x260": 56.2, "ms_640x480": 130.8},
        "MSFET-E2V": {"params_m": 16.71, "ms_240x180": 16.8, "ms_340x260": 26.4, "ms_640x480": 35.73},
    }


def table_ablation_modules() -> dict[str, dict[str, dict[str, float]]]:
    """Table IV — component ablation on ECD / HQF / MVSEC."""
    return {
        "Model I": {
            "ECD": {"psnr": 10.35, "ssim": 0.465, "lpips": 0.355},
            "HQF": {"psnr": 11.35, "ssim": 0.462, "lpips": 0.384},
            "MVSEC": {"psnr": 6.58, "ssim": 0.244, "lpips": 0.578},
        },
        "Model II": {
            "ECD": {"psnr": 12.44, "ssim": 0.510, "lpips": 0.298},
            "HQF": {"psnr": 13.17, "ssim": 0.485, "lpips": 0.321},
            "MVSEC": {"psnr": 8.48, "ssim": 0.264, "lpips": 0.533},
        },
        "Model III": {
            "ECD": {"psnr": 13.27, "ssim": 0.534, "lpips": 0.286},
            "HQF": {"psnr": 14.23, "ssim": 0.517, "lpips": 0.289},
            "MVSEC": {"psnr": 10.05, "ssim": 0.297, "lpips": 0.521},
        },
        "MSFET-E2V": {
            "ECD": {"psnr": 13.98, "ssim": 0.584, "lpips": 0.252},
            "HQF": {"psnr": 14.82, "ssim": 0.548, "lpips": 0.291},
            "MVSEC": {"psnr": 11.42, "ssim": 0.362, "lpips": 0.501},
        },
    }


def table_ablation_cdam_subbands() -> dict[str, dict[str, dict[str, float]]]:
    """Table V — CDAM LL vs HF vs full."""
    return {
        "CDAM LL": {
            "ECD": {"psnr": 10.35, "ssim": 0.499, "lpips": 0.322},
            "HQF": {"psnr": 11.35, "ssim": 0.462, "lpips": 0.344},
            "MVSEC": {"psnr": 7.58, "ssim": 0.299, "lpips": 0.578},
        },
        "CDAM HF": {
            "ECD": {"psnr": 12.44, "ssim": 0.543, "lpips": 0.283},
            "HQF": {"psnr": 13.17, "ssim": 0.485, "lpips": 0.289},
            "MVSEC": {"psnr": 9.48, "ssim": 0.324, "lpips": 0.533},
        },
        "MSFET-E2V": {
            "ECD": {"psnr": 13.98, "ssim": 0.584, "lpips": 0.252},
            "HQF": {"psnr": 14.82, "ssim": 0.548, "lpips": 0.291},
            "MVSEC": {"psnr": 11.42, "ssim": 0.362, "lpips": 0.502},
        },
    }


def table_ablation_depth() -> dict[str, dict[str, dict[str, float]]]:
    """Table VI — encoder-decoder depth d."""
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for d, ecd, hqf, mv in [
        (1, (8.19, 0.399, 0.302), (10.56, 0.362, 0.396), (7.86, 0.218, 0.605)),
        (2, (10.35, 0.496, 0.286), (11.96, 0.496, 0.324), (9.86, 0.299, 0.558)),
        (3, (13.98, 0.584, 0.252), (14.82, 0.548, 0.291), (11.42, 0.362, 0.501)),
        (4, (10.25, 0.581, 0.256), (15.32, 0.526, 0.298), (11.42, 0.358, 0.504)),
    ]:
        rows[f"d={d}"] = {
            "ECD": {"psnr": ecd[0], "ssim": ecd[1], "lpips": ecd[2]},
            "HQF": {"psnr": hqf[0], "ssim": hqf[1], "lpips": hqf[2]},
            "MVSEC": {"psnr": mv[0], "ssim": mv[1], "lpips": mv[2]},
        }
    return rows


def table_ablation_voxel_bins() -> dict[int, dict[str, dict[str, float]]]:
    """Table VII — temporal bins b (ECD/HQF/MVSEC SSIM at b=5 is paper default)."""
    return {
        2: {"ECD": {"ssim": 0.522}, "HQF": {"ssim": 0.498}, "MVSEC": {"ssim": 0.298}},
        5: {"ECD": {"ssim": 0.584}, "HQF": {"ssim": 0.548}, "MVSEC": {"ssim": 0.362}},
        10: {"ECD": {"ssim": 0.572}, "HQF": {"ssim": 0.541}, "MVSEC": {"ssim": 0.342}},
    }


def training_step_demo(cfg: MSFETE2VConfig | None = None) -> dict[str, float]:
    """Smoke: voxel grid -> MSFET-E2V forward + combined loss."""
    cfg = cfg or MSFETE2VConfig()
    torch.manual_seed(4)
    h, w, b = 64, 64, cfg.voxel_bins
    n_ev = 400
    x = torch.randint(0, w, (n_ev,))
    y = torch.randint(0, h, (n_ev,))
    t = torch.rand(n_ev)
    p = torch.where(torch.rand(n_ev) > 0.5, torch.ones(n_ev), -torch.ones(n_ev))
    vox = events_to_voxel(x, y, t, p, height=h, width=w, bins=b)
    batch = vox.unsqueeze(0)
    target = torch.rand(1, 1, h, w)
    model = MSFETE2V(cfg)
    pred = model(batch)
    loss, parts = total_loss(pred.unsqueeze(0), target.unsqueeze(0))
    return {
        "pred_shape_h": float(pred.shape[-2]),
        "loss": float(loss.detach()),
        **parts,
        "voxel_nonzero": float((vox != 0).sum().item()),
    }


def evaluation_demo(cfg: MSFETE2VConfig | None = None) -> dict[str, Any]:
    """Smoke: paper table ordering + DWT roundtrip + proxy metrics on synthetic pair."""
    cfg = cfg or MSFETE2VConfig()
    step = training_step_demo(cfg)
    torch.manual_seed(7)
    x = torch.randn(1, 8, 32, 32)
    ll, lh, hl, hh = haar_dwt2d(x)
    recon = haar_idwt2d(ll, lh, hl, hh)
    dwt_mse = float(F.mse_loss(x, recon).item())

    tab = table_quantitative_e2v()
    ours_ecd = tab["ECD"]["MSFET-E2V"]
    et_ecd = tab["ECD"]["ET-Net"]
    eff = table_inference_efficiency()
    abl = table_ablation_modules()

    pred = torch.rand(1, 1, 32, 32)
    tgt = pred + torch.randn_like(pred) * 0.05
    proxy_psnr = psnr(pred, tgt)
    proxy_ssim = ssim_proxy(pred, tgt)

    depth = table_ablation_depth()
    bins = table_ablation_voxel_bins()

    return {
        **step,
        "dwt_roundtrip_mse": dwt_mse,
        "msfet_beats_etnet_ssim_ecd": ours_ecd["ssim"] > et_ecd["ssim"],
        "msfet_beats_etnet_lpips_ecd": ours_ecd["lpips"] < et_ecd["lpips"],
        "etnet_slower_than_msfet_640": eff["ET-Net"]["ms_640x480"] > eff["MSFET-E2V"]["ms_640x480"],
        "ablation_monotonic_ssim_ecd": (
            abl["Model I"]["ECD"]["ssim"]
            < abl["Model II"]["ECD"]["ssim"]
            < abl["Model III"]["ECD"]["ssim"]
            < abl["MSFET-E2V"]["ECD"]["ssim"]
        ),
        "optimal_depth_d3_ssim_ecd": depth["d=3"]["ECD"]["ssim"] >= depth["d=4"]["ECD"]["ssim"],
        "optimal_bins_b5": bins[5]["ECD"]["ssim"] >= bins[10]["ECD"]["ssim"],
        "proxy_psnr": proxy_psnr,
        "proxy_ssim": proxy_ssim,
        "brisque_msfet_best_night": table_brisque()["MSFET-E2V"]["mvsec_night"]
        < table_brisque()["ET-Net"]["mvsec_night"],
    }
