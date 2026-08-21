"""EIGENET framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.eigenet.benchmarks import (
    TABLE2_AR_EIGENET,
    TABLE2_AR_XRIR,
    TABLE3_HAA_DAMPENED_T60,
    benchmarks_bundle,
)
from ltx_trainer.eigenet.config import EigeNetConfig
from ltx_trainer.eigenet.cvat import cvat_forward, stack_view_tokens
from ltx_trainer.eigenet.metrics import metric_triplet
from ltx_trainer.eigenet.modulation import geometry_informed_modulate, spectrum_loss, total_training_loss


def framework_card(cfg: EigeNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EigeNetConfig()
    return {
        "name": "EIGENET",
        "paper": cfg.paper_arxiv,
        "title": "Geometry-Informed Multi-Modal Learning for Few-shot Novel View RIR Prediction",
        "task": "few_shot_novel_view_rir",
        "architecture": {
            "cvat_blocks": cfg.cvat_blocks,
            "cvat_dim": cfg.cvat_dim,
            "geom_encoder": "ViT on 6-ch depth propagation tensor + coordinate MLP",
            "acoustic_encoder": "frozen DAC 16 kHz latents (reference); SPE+modulation (target)",
            "modulation": "DiT-style geometry-informed block + 7-band octave spectrum aux loss",
            "n_parameters_m": cfg.n_parameters_m,
        },
        "datasets": list(cfg.datasets),
        "github": cfg.github,
    }


def knowledge_card(cfg: EigeNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EigeNetConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_results": {
            "ar_k1_edt_eigenet": TABLE2_AR_EIGENET[1]["EDT"],
            "ar_k1_edt_xrir": TABLE2_AR_XRIR[1]["EDT"],
            "ar_k8_t60_eigenet": TABLE2_AR_EIGENET[8]["T60"],
            "haa_dampened_t60_k1_xrir": TABLE3_HAA_DAMPENED_T60["xRIR"][1],
            "haa_dampened_t60_k1_ours": TABLE3_HAA_DAMPENED_T60["Ours"][1],
            "dac_reconstruction": bench["table1_dac_ar"],
        },
        "design_notes": {
            "alternate_attention": "Alternates local intra-view and global cross-view attention (CVAT).",
            "modulation_block": "Geometry modulates target acoustic tokens; auxiliary 7-band spectrum loss.",
            "masking_probe": "AA needs both geometric and acoustic tokens; CA/SA interpolate acoustics.",
        },
        "integration": {
            "env": "GOPEX_EIGENET=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,eigenet",
            "cli": "./scripts/gopex-eigenet.sh",
        },
    }


def evaluation_demo(cfg: EigeNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EigeNetConfig()
    rng = np.random.default_rng(42)
    sr = cfg.sampling_rate_hz
    n = cfg.rir_length_samples
    target = rng.standard_normal(n) * 0.1
    target[n // 10 :] *= np.exp(-np.linspace(0, 4, n - n // 10))
    pred = target + rng.standard_normal(n) * 0.02

    geom = rng.standard_normal(cfg.geom_feature_dim)
    proxy = rng.standard_normal(cfg.n_acoustic_tokens * cfg.acoustic_token_dim).reshape(
        cfg.n_acoustic_tokens, cfg.acoustic_token_dim
    )[:, 0]
    a0, s_hat = geometry_informed_modulate(geom, proxy, cfg=cfg)
    s_tgt = np.abs(rng.standard_normal(cfg.n_octave_bands))
    spec = spectrum_loss(s_hat, s_tgt, cfg=cfg)
    loss = total_training_loss(0.5, spec["L_spectrum"], cfg=cfg)

    feat = min(8, cfg.n_acoustic_tokens)
    g_tok = geom.reshape(1, -1)[:, :feat]
    a_tok = a0.reshape(-1, 1)[:, :feat]
    p_tok = proxy.reshape(-1, 1)[:, :feat]
    v0 = stack_view_tokens(g_tok, a_tok)
    v1 = stack_view_tokens(g_tok, p_tok)
    h = np.concatenate([v0, v1], axis=0)
    _ = cvat_forward(h, n_views=2, n_blocks=2)

    metrics = metric_triplet(pred, target, sr=sr)
    return {
        "synthetic_rir_length": n,
        "spectrum_loss": spec,
        "total_loss_stub": loss,
        "metrics": metrics,
        "paper_tables": {
            "eigenet_beats_xrir_ar_k1_edt": TABLE2_AR_EIGENET[1]["EDT"] < TABLE2_AR_XRIR[1]["EDT"],
            "eigenet_beats_xrir_haa_dampened_t60_k1": TABLE3_HAA_DAMPENED_T60["Ours"][1]
            < TABLE3_HAA_DAMPENED_T60["xRIR"][1],
        },
        "reference_counts": list(TABLE2_AR_EIGENET.keys()),
    }


def evaluation_smoke(cfg: EigeNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EigeNetConfig()
    ev = evaluation_demo(cfg)
    return {
        "package": "ltx_trainer.eigenet",
        "status": "smoke_ok",
        "paper": cfg.paper_arxiv,
        "n_parameters_m": cfg.n_parameters_m,
        "metrics_finite": all(np.isfinite(v) for v in ev["metrics"].values()),
        "paper_sota_stub": ev["paper_tables"]["eigenet_beats_xrir_ar_k1_edt"],
    }
