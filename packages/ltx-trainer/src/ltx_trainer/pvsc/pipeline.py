"""PVSC framework card and paper tables (arXiv:2605.19397)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.pvsc.channel import awgn_receive, channel_bandwidth_ratio, zf_equalize
from ltx_trainer.pvsc.config import PVSCConfig
from ltx_trainer.pvsc.entropy import DEFAULT_RATE_SET, quantize_rate, symbol_length_factor
from ltx_trainer.pvsc.layout import LIMITATIONS
from ltx_trainer.pvsc.loss import overall_objective, rate_loss


def framework_card(cfg: PVSCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PVSCConfig()
    return {
        "name": "PVSC",
        "paper": cfg.paper_arxiv,
        "full_title": "Perception-Aware Video Semantic Communication",
        "idea": (
            "End-to-end semantic video comms: no explicit motion vectors; spatio-temporal "
            "feature coding + entropy-based rate matching + DPO-free RL-free perceptual training."
        ),
        "stages": [
            "Stage 1: ideal decoding branch (semantic + entropy + GAN)",
            "Stage 2: channel-aware Ftx/Frx symbol mapping",
            "Stage 3: joint fine-tune + rate-set distillation",
        ],
        "streams_per_frame": ["complex symbols st", "hyperprior bz_t", "rate map bk_t"],
        "metrics": ["LPIPS", "DISTS", "BD-CBR vs VTM+5G LDPC"],
        "realtime_rtx4090_fps": {
            "1080p": "22.7 / 37.0 (Tx/Rx)",
            "720p": "45.0 / 74.1",
            "480p": "51.8 / 90.9",
        },
        "defaults": cfg.__dict__,
    }


def table_i_bd_cbr_awgn_snr6() -> list[dict[str, Any]]:
    """Table I excerpt — BD-CBR (%) LPIPS/DISTS vs VTM+5G LDPC, GOP=4, average."""
    return [
        {"method": "VTM-17.0 + 5G LDPC", "lpips": 0.0, "dists": 0.0},
        {"method": "DCVC-RT + 5G LDPC", "lpips": 23.0, "dists": 44.4},
        {"method": "GLC-Video + 5G LDPC", "lpips": -70.5, "dists": -84.9},
        {"method": "PVSC (Ours)", "lpips": -82.0, "dists": -88.9},
    ]


def table_i_pvsc_by_gop() -> list[dict[str, Any]]:
    """Table I — PVSC rows across GOP sizes (average LPIPS/DISTS BD-CBR)."""
    return [
        {"gop": 4, "lpips": -82.0, "dists": -88.9},
        {"gop": 8, "lpips": -79.8, "dists": -88.8},
        {"gop": 12, "lpips": -77.3, "dists": -87.9},
    ]


def table_iv_complexity() -> list[dict[str, Any]]:
    """Table IV — RTX 4090 latency and MACs/pixel (1080p excerpt)."""
    return [
        {
            "method": "PVSC (Ours)",
            "tx_ms": 44.1,
            "rx_ms": 27.0,
            "macs_m_per_pixel": 0.41,
            "params_m": 35.4,
        },
        {
            "method": "DVST",
            "tx_ms": 351.8,
            "rx_ms": 122.1,
            "macs_m_per_pixel": 2.77,
            "params_m": 48.4,
        },
        {
            "method": "GLC-Video",
            "tx_ms": 167.8,
            "rx_ms": 266.9,
            "macs_m_per_pixel": 3.28,
            "params_m": 501.2,
        },
    ]


def table_vi_loss_ablation() -> list[dict[str, Any]]:
    """Table VI — loss term ablation on UVG (BD-CBR %)."""
    return [
        {"method": "PVSC", "Lrec": True, "Lper": False, "Ladv": False, "lpips": 41.3, "dists": 45.4},
        {"method": "PVSC", "Lrec": True, "Lper": True, "Ladv": False, "lpips": -75.3, "dists": -91.0},
        {"method": "PVSC", "Lrec": True, "Lper": True, "Ladv": True, "lpips": -78.1, "dists": -92.2},
    ]


def headline_savings() -> dict[str, str]:
    return {
        "lpips_bandwidth_saving": "up to ~75% vs VTM+5G LDPC at comparable LPIPS",
        "dists_bandwidth_saving": "up to ~87% vs VTM+5G LDPC at comparable DISTS",
        "vbench_avg_note": "competitive perceptual video quality vs separated codecs",
    }


def pipeline_demo(cfg: PVSCConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or PVSCConfig()
    rng = np.random.default_rng(seed)
    symbols = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    rx = awgn_receive(symbols, snr_db=cfg.eval_snr_db)
    eq = zf_equalize(rx, channel_est=1.0 + 0.05j)

    k_raw = symbol_length_factor(0.35, eta=cfg.eta_spectral, channels=cfg.feature_channels)
    k_q = quantize_rate(k_raw, DEFAULT_RATE_SET)

    cbr = channel_bandwidth_ratio(
        channel_uses=int(k_q) * 64,
        frames=cfg.gop_length,
        height=256,
        width=256,
    )

    obj = overall_objective(
        rec_losses=[0.02],
        per_losses=[0.15],
        adv_losses=[0.05],
        rate=rate_loss([k_q], rate_map_bits=128, hyperprior_bits=256, eta=cfg.eta_spectral),
        lambda_rec=cfg.lambda_rec,
        lambda_per=cfg.lambda_per,
        lambda_adv=cfg.lambda_adv,
    )

    return {
        "symbol_length_raw": k_raw,
        "symbol_length_quantized": k_q,
        "cbr_toy": cbr,
        "zf_output_norm": float(np.linalg.norm(eq)),
        "objective_scalar": obj,
        "rate_set": list(DEFAULT_RATE_SET),
    }


def evaluation_demo(cfg: PVSCConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["headline"] = headline_savings()
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_excerpt": table_i_bd_cbr_awgn_snr6(),
        "table_i_pvsc_by_gop": table_i_pvsc_by_gop(),
        "table_iv_complexity": table_iv_complexity(),
        "table_vi_loss_ablation": table_vi_loss_ablation(),
        "headline_savings": headline_savings(),
        "training": {
            "datasets": ["Vimeo-90K", "BVI-DVC"],
            "crop": "256x256",
            "channel_train_snr_db": 10,
        },
    }
