"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dbhn_net.config import DbhnNetConfig
from ltx_trainer.dbhn_net.lif import gradient_proxy, lif_step
from ltx_trainer.dbhn_net.loss import combined_loss
from ltx_trainer.dbhn_net.modules import (
    band_merge_bands,
    band_split_spectrum,
    information_transformation_block,
    interaction_block,
    tf_cross_attention_fusion,
)


def framework_card(cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    c = cfg or DbhnNetConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "monaural_speech_enhancement",
        "branches": ["ANN (BandSplit + TF-Mamba)", "SNN (SFEG + ITB)"],
        "fusion": ["Interaction (3 stages)", "TF-Cross Attention-Fusion"],
        "headline": headline_results(c),
    }


def headline_results(cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    c = cfg or DbhnNetConfig()
    return {
        "wsj0_pesq": c.wsj0_pesq_avg,
        "wsj0_estoi": c.wsj0_estoi_avg,
        "wsj0_sisdr": c.wsj0_sisdr_avg,
        "dns_wb_pesq": c.dns_wb_pesq,
        "dns_sisdr": c.dns_sisdr,
        "macs_gps": c.macs_dbhn,
        "complexity_reduction_x": c.complexity_reduction_x,
        "dual_branch_beats_single": c.full_pesq > c.wo_ann_pesq and c.full_pesq > c.wo_snn_pesq,
    }


def table2_dual_branch(cfg: DbhnNetConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or DbhnNetConfig()
    return [
        {
            "model": "DBHN-Net (OURS)",
            "pesq": c.full_pesq,
            "estoi": c.full_estoi,
            "sisdr": c.full_sisdr,
        },
        {"model": "w/o ann-branch", "pesq": c.wo_ann_pesq, "estoi": 74.91, "sisdr": 10.41},
        {"model": "w/o snn-branch", "pesq": c.wo_snn_pesq, "estoi": 77.32, "sisdr": 10.89},
    ]


def table3_mamba_ablation(cfg: DbhnNetConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or DbhnNetConfig()
    return [
        {
            "model": "DBHN-Net (OURS)",
            "pesq": c.full_pesq,
            "estoi": c.full_estoi,
            "sisdr": c.full_sisdr,
            "macs": c.macs_dbhn,
        },
        {"model": "LSTM", "pesq": 3.01, "estoi": 79.09, "sisdr": 11.18, "macs": c.macs_lstm},
        {
            "model": "Transformer",
            "pesq": 3.13,
            "estoi": 82.12,
            "sisdr": 11.97,
            "macs": c.macs_transformer,
        },
    ]


def table9_complexity(cfg: DbhnNetConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or DbhnNetConfig()
    return [
        {"model": "BSDBNet", "macs": c.macs_bsdb},
        {"model": "GaG-Net", "macs": c.macs_gag},
        {"model": "DBHN-Net", "macs": c.macs_dbhn},
    ]


def table6_wsj0(cfg: DbhnNetConfig | None = None) -> dict[str, float]:
    c = cfg or DbhnNetConfig()
    return {"pesq_avg": c.wsj0_pesq_avg, "estoi_avg": c.wsj0_estoi_avg, "sisdr_avg": c.wsj0_sisdr_avg}


def table7_voicebank(cfg: DbhnNetConfig | None = None) -> dict[str, float]:
    c = cfg or DbhnNetConfig()
    return {
        "wb_pesq": c.vb_wb_pesq,
        "stoi": c.vb_stoi,
        "csig": c.vb_csig,
        "cbak": c.vb_cbak,
        "covl": c.vb_covl,
    }


def table8_dns(cfg: DbhnNetConfig | None = None) -> dict[str, float]:
    c = cfg or DbhnNetConfig()
    return {
        "wb_pesq": c.dns_wb_pesq,
        "nb_pesq": c.dns_nb_pesq,
        "stoi": c.dns_stoi,
        "sisdr": c.dns_sisdr,
    }


def benchmarks_bundle(cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    c = cfg or DbhnNetConfig()
    return {
        "table2_dual_branch": table2_dual_branch(c),
        "table3_mamba": table3_mamba_ablation(c),
        "table6_wsj0": table6_wsj0(c),
        "table7_voicebank": table7_voicebank(c),
        "table8_dns": table8_dns(c),
        "table9_complexity": table9_complexity(c),
    }


def pipeline_demo(seed: int = 42, cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    """CPU forward stub: band-split → dual branch → TF-CAF → loss."""
    c = cfg or DbhnNetConfig()
    rng = np.random.default_rng(seed)
    t, f = 32, c.freq_bins
    noisy_r = rng.standard_normal((t, f))
    noisy_i = rng.standard_normal((t, f))
    clean_r = noisy_r * 0.7
    clean_i = noisy_i * 0.7

    bands = band_split_spectrum(noisy_r, n_bands=4)
    ann_feat = band_merge_bands(bands)
    membrane = np.zeros_like(noisy_r)
    spikes, membrane = lif_step(noisy_r, membrane)
    snn_feat, _ = interaction_block(ann_feat, spikes)
    snn_refined = information_transformation_block(snn_feat)
    enhanced = tf_cross_attention_fusion(ann_feat, snn_refined)

    loss = combined_loss(enhanced, noisy_i * 0.5, clean_r, clean_i, beta=c.loss_beta)
    proxy = float(np.mean(gradient_proxy(spikes, alpha=c.lif_alpha)))

    return {
        "spectrum_shape": [t, f],
        "n_bands": len(bands),
        "n_tf_mamba_blocks": c.tf_mamba_blocks,
        "loss": loss,
        "lif_spike_rate": float(np.mean(spikes)),
        "gradient_proxy_mean": proxy,
        "macs_gps": c.macs_dbhn,
    }


def evaluation_demo(seed: int = 42, cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(seed=seed, cfg=cfg)
    c = cfg or DbhnNetConfig()
    return {
        **demo,
        "pesq_anchor": c.full_pesq,
        "complexity_reduction_x": c.complexity_reduction_x,
    }
