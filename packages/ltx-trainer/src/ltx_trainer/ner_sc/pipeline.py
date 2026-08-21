"""NeR-SC framework card, paper tables, and evaluation demo (arXiv:2605.27024)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ner_sc.config import NeRSCConfig
from ltx_trainer.ner_sc.layout import LIMITATIONS
from ltx_trainer.ner_sc.metrics import bpp_from_param_count, ms_ssim, psnr_db
from ltx_trainer.ner_sc.mgf import mgf_fuse
from ltx_trainer.ner_sc.palette import (
    augment_palette_from_ll,
    init_screen_palette,
    logits_nearest_palette,
    reconstruct_ll_from_palette,
)
from ltx_trainer.ner_sc.skip import decode_with_skip, normalized_embedding_l1
from ltx_trainer.ner_sc.wavelet import haar_dwt2_rgb, haar_idwt2_rgb


def framework_card(cfg: NeRSCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NeRSCConfig()
    return {
        "name": "NeR-SC",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "backbone": cfg.backbone,
        "idea": (
            "Screen-content neural video representation: SNeRV Haar LL/HF split + learnable color palette "
            "(LL only), multi-gate dense fusion (F3–F5), embedding-level frame skip (τ=0.005)."
        ),
        "modules": [
            "Learnable color palette (K=64, temperature softmax on LL)",
            "Multi-gate dense fusion + SE residual gating",
            "Embedding-level skip (zero training overhead)",
        ],
        "datasets": ["DSCVC (videos 11–20)", "VCD (6 sequences, 1800 frames)"],
        "crop": f"{cfg.crop_resolution[0]}×{cfg.crop_resolution[1]}",
        "headline_psnr_db": {"DSCVC": cfg.psnr_dscvc_avg_db, "VCD": cfg.psnr_vcd_avg_db},
        "decode_fps_with_skip": cfg.decode_fps_with_skip,
        "defaults": cfg.__dict__,
    }


def table_i_video16_model_sizes() -> dict[str, list[dict[str, Any]]]:
    """Table I(a) — Video 16 PSNR/MS-SSIM vs model size (1.5M–6.0M)."""
    return {
        "model_size_m": [
            {"method": "NeRV", "1.5M": "28.08/.9052", "3.0M": "30.21/.9380", "4.5M": "31.91/.9558", "6.0M": "33.62/.9686"},
            {"method": "HNeRV", "1.5M": "30.46/.9381", "3.0M": "34.16/.9681", "4.5M": "35.51/.9752", "6.0M": "36.81/.9803"},
            {"method": "SNeRV", "1.5M": "38.11/.9880", "3.0M": "40.91/.9927", "4.5M": "41.79/.9940", "6.0M": "41.92/.9940"},
            {"method": "NeR-SC", "1.5M": "38.54/.9888", "3.0M": "41.12/.9929", "4.5M": "42.37/.9944", "6.0M": "43.33/.9956"},
        ],
        "epochs": [
            {"method": "SNeRV", "300": "40.91/.9927", "600": "42.04/.9944", "1200": "42.73/.9950", "1800": "42.96/.9958"},
            {"method": "NeR-SC", "300": "41.12/.9929", "600": "42.17/.9945", "1200": "42.84/.9955", "1800": "43.13/.9958"},
        ],
    }


def table_ii_per_sequence() -> list[dict[str, Any]]:
    """Table II — per-sequence PSNR/MS-SSIM on DSCVC + VCD."""
    rows = [
        ("Video 11", "38.34/.9992", "38.58/.9993"),
        ("Video 12", "36.60/.9967", "36.87/.9967"),
        ("Video 13", "31.22/.9735", "31.49/.9755"),
        ("Video 14", "38.67/.9972", "38.89/.9975"),
        ("Video 15", "39.60/.9956", "39.90/.9959"),
        ("Video 16", "40.91/.9927", "41.07/.9928"),
        ("Video 17", "41.10/.9917", "41.93/.9928"),
        ("Video 18", "37.25/.9918", "37.79/.9929"),
        ("Video 19", "49.59/.9994", "49.79/.9995"),
        ("Video 20", "44.47/.9984", "46.87/.9988"),
        ("th001", "35.76/.9918", "35.78/.9920"),
        ("thbb001", "43.61/.9906", "44.34/.9922"),
        ("thob001", "40.47/.9929", "40.40/.9927"),
    ]
    return [
        {
            "video": name,
            "snerv_psnr_msssim": snerv,
            "ner_sc_psnr_msssim": nersc,
        }
        for name, snerv, nersc in rows
    ]


def table_iii_ablation() -> list[dict[str, Any]]:
    """Table III — component ablation (PSNR dB / FPS)."""
    return [
        {"config": "Baseline", "video_16": "38.14/39.9", "video_17": "38.42/41.7", "video_18": "33.48/41.5", "avg_psnr": 36.68, "avg_fps": 41.0},
        {"config": "+ Palette", "video_16": "38.28/36.0", "video_17": "38.56/35.7", "video_18": "33.85/36.8", "avg_psnr": 36.90, "avg_fps": 36.1, "delta_psnr": 0.22},
        {"config": "+ MGF", "video_16": "38.37/41.6", "video_17": "38.97/41.8", "video_18": "33.88/41.9", "avg_psnr": 37.07, "avg_fps": 41.8, "delta_psnr": 0.39},
        {"config": "+ Skip", "video_16": "38.37/56.1", "video_17": "38.97/71.6", "video_18": "33.88/57.4", "avg_psnr": 37.07, "avg_fps": 61.7, "delta_fps": 20.7},
    ]


def table_iv_palette_size() -> list[dict[str, Any]]:
    """Table IV — palette size K ablation (1.5M params, 100 epochs)."""
    return [
        {"video": "video 16", "k16": "32.84/0.973", "k32": "32.49/0.971", "k64": "33.11/0.975", "k128": "32.65/0.973"},
        {"video": "video 17", "k16": "33.18/0.973", "k32": "32.89/0.973", "k64": "32.14/0.971", "k128": "32.25/0.971"},
        {"video": "video 18", "k16": "22.36/0.881", "k32": "23.54/0.906", "k64": "23.78/0.909", "k128": "22.89/0.884"},
        {"video": "Avg.", "k16": "29.46/0.942", "k32": "29.64/0.950", "k64": "29.68/0.952", "k128": "29.26/0.943"},
    ]


def rd_curve_anchor_points() -> dict[str, list[dict[str, float]]]:
    """Fig. 2 — qualitative RD anchors (PSNR vs bpp) for plotting."""
    return {
        "DSCVC": [
            {"bpp": 0.02, "NeRV": 27.5, "HNeRV": 32.0, "SNeRV": 38.5, "NeR-SC": 39.0, "H264": 36.0, "H265": 37.0},
            {"bpp": 0.08, "NeRV": 33.0, "HNeRV": 36.5, "SNeRV": 41.0, "NeR-SC": 41.5, "H264": 39.0, "H265": 39.5},
            {"bpp": 0.14, "NeRV": 35.0, "HNeRV": 38.0, "SNeRV": 42.0, "NeR-SC": 42.8, "H264": 40.0, "H265": 40.5},
        ],
        "VCD": [
            {"bpp": 0.02, "NeRV": 34.0, "HNeRV": 37.0, "SNeRV": 40.0, "NeR-SC": 40.5, "H264": 38.5, "H265": 39.0},
            {"bpp": 0.06, "NeRV": 38.0, "HNeRV": 40.0, "SNeRV": 41.5, "NeR-SC": 42.0, "H264": 40.5, "H265": 41.0},
            {"bpp": 0.09, "NeRV": 39.0, "HNeRV": 41.0, "SNeRV": 42.0, "NeR-SC": 42.5, "H264": 41.0, "H265": 41.5},
        ],
    }


def skip_threshold_tradeoff() -> list[dict[str, Any]]:
    """Fig. 4 — embedding skip τ vs FPS (video 16 excerpt)."""
    return [
        {"tau": "baseline", "fps": 42.4, "psnr_db": 40.96, "skip_rate_pct": 0.0},
        {"tau": 0.001, "fps": 53.9, "psnr_db": 40.96, "skip_rate_pct": 15.4},
        {"tau": 0.005, "fps": 62.6, "psnr_db": 40.96, "skip_rate_pct": 28.2},
        {"tau": 0.05, "fps": 61.8, "psnr_db": 40.70, "skip_rate_pct": 28.2, "note": "mild loss on video 16"},
    ]


def _synthetic_screen_frame(h: int, w: int, *, seed: int) -> np.ndarray:
    """UI-like frame: flat regions + sharp edges."""
    rng = np.random.default_rng(seed)
    frame = np.ones((h, w, 3), dtype=np.float64) * 0.92
    frame[: h // 4, :] = 0.15
    frame[h // 4 : h // 4 + 20, :] = 0.0
    frame[:, : w // 3] = 0.85
    # Syntax colors.
    for _ in range(8):
        y, x = rng.integers(h // 4, h - 20), rng.integers(w // 3, w - 10)
        color = rng.choice(
            [
                [0.0, 0.45, 0.85],
                [0.85, 0.2, 0.2],
                [0.1, 0.65, 0.35],
            ]
        )
        frame[y : y + 12, x : x + 40] = color
    return np.clip(frame, 0.0, 1.0)


def pipeline_demo(cfg: NeRSCConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    """End-to-end stub: Haar → palette LL + HF conv → IDWT; MGF; skip on embeddings."""
    cfg = cfg or NeRSCConfig()
    h, w = 64, 64
    gt = _synthetic_screen_frame(h, w, seed=seed)
    ll, lh, hl, hh = haar_dwt2_rgb(gt)
    k = cfg.palette_size_k
    rng = np.random.default_rng(seed)
    palette = augment_palette_from_ll(ll, init_screen_palette(k, seed=seed))
    logits = logits_nearest_palette(ll, palette)
    ll_hat = reconstruct_ll_from_palette(logits, palette, temperature=cfg.palette_temperature)
    # HF stub: copy ground-truth HF bands (trained heads in full model).
    recon = haar_idwt2_rgb(ll_hat, lh, hl, hh)
    psnr = psnr_db(gt, recon)

    # MGF on three scale feature maps derived from LL.
    f3 = ll.transpose(2, 0, 1)
    f4 = np.mean(f3, axis=0, keepdims=True)
    f4 = np.repeat(f4, f3.shape[0], axis=0)
    f5 = f3 * 0.95
    mgf_out = mgf_fuse([f3, f4, f5], fuse_channels=cfg.mgf_fuse_channels)

    # Skip: static embeddings for most frames.
    emb_static = rng.standard_normal(16 * 2 * 4)
    emb_move = emb_static.copy()
    emb_move[0] += 0.5
    embeddings = [emb_static, emb_static, emb_static, emb_move, emb_static]

    def _decode(z: np.ndarray) -> np.ndarray:
        _ = z
        return recon

    _, skip_stats = decode_with_skip(embeddings, _decode, tau=cfg.skip_threshold_tau)
    bpp = bpp_from_param_count(
        int(cfg.default_params_m * 1e6),
        n_frames=len(embeddings),
        height=h,
        width=w,
    )
    return {
        "psnr_db": round(psnr, 3),
        "ms_ssim": round(ms_ssim(gt, recon), 4),
        "bpp_proxy": round(bpp, 5),
        "mgf_channels": int(mgf_out.shape[0]),
        "skip": skip_stats,
        "embedding_l1_static": round(normalized_embedding_l1(emb_static, emb_static), 6),
    }


def benchmarks_bundle() -> dict[str, Any]:
    cfg = NeRSCConfig()
    return {
        "table_i": table_i_video16_model_sizes(),
        "table_ii": table_ii_per_sequence(),
        "table_iii_ablation": table_iii_ablation(),
        "table_iv_palette": table_iv_palette_size(),
        "rd_curves": rd_curve_anchor_points(),
        "skip_tradeoff": skip_threshold_tradeoff(),
        "averages": {
            "DSCVC_psnr_db": cfg.psnr_dscvc_avg_db,
            "VCD_psnr_db": cfg.psnr_vcd_avg_db,
        },
    }


def evaluation_demo(cfg: NeRSCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NeRSCConfig()
    return {
        "framework": framework_card(cfg),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "paper_tables": benchmarks_bundle(),
    }
