"""COMET framework card and paper tables (arXiv:2605.29628)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.comet.config import CometConfig
from ltx_trainer.comet.gap import cosine_sim, embedding_shift, modality_gap_sources
from ltx_trainer.comet.layout import LIMITATIONS
from ltx_trainer.comet.pls import (
    covariance_decomposition,
    net_useful_contribution,
    pls_svd,
    project_coefficients,
    similarity_direct_cross,
)
from ltx_trainer.comet.plshead import head_energy_ratio, linear_projection_decoding, plshead_truncate
from ltx_trainer.comet.retrieval import retrieval_smoke


def framework_card(cfg: CometConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CometConfig()
    return {
        "name": "COMET",
        "paper": cfg.paper_arxiv,
        "idea": (
            "PLS-SVD dissects CLAP embeddings into mean (static gap), compact shared semantic head "
            "(~100 dims), and modality-private tail. PLSHead spectral truncation mitigates the gap "
            "training-free and compresses 1024→100 dims for retrieval and zero-shot captioning."
        ),
        "decomposition": {
            "method": "PLS-SVD on M = T^T A",
            "embed_dim": cfg.embed_dim,
            "head_size": cfg.head_size,
            "clap_model": cfg.clap_model,
        },
        "gap_sources": modality_gap_sources(),
        "plshead": {
            "training": "top-K text projection coefficients t100",
            "inference": "top-K audio projection coefficients a100",
            "memory": "202 vectors (means + directions), no PD memory bank",
        },
        "limitations": list(LIMITATIONS),
    }


def table_i_contribution() -> list[dict[str, Any]]:
    """Table I — direct vs cross similarity contributions (Clotho test)."""
    cfg = CometConfig()
    return [
        {
            "pair_type": "positive",
            "contrib_direct": cfg.contrib_direct_pos,
            "contrib_direct_100": cfg.contrib_direct100_pos,
            "contrib_cross": cfg.contrib_cross_pos,
        },
        {
            "pair_type": "negative",
            "contrib_direct": cfg.contrib_direct_neg,
            "contrib_direct_100": cfg.contrib_direct_neg,
            "contrib_cross": cfg.contrib_cross_neg,
        },
    ]


def table_iii_retrieval_clotho() -> list[dict[str, Any]]:
    """Table III excerpt — Clotho in-domain text→audio."""
    return [
        {"method": "Original", "R1": 17.42, "R5": 39.64, "R10": 52.19, "mAP10": 27.02},
        {"method": "PLSHead", "R1": 17.32, "R5": 41.21, "R10": 54.05, "mAP10": 27.56},
        {"method": "PLSHeadW", "R1": 17.30, "R5": 41.11, "R10": 54.30, "mAP10": 27.52},
        {"method": "PCAHead", "R1": 0.06, "R5": 0.42, "R10": 0.82, "mAP10": 0.23},
    ]


def table_iii_retrieval_audiocaps() -> list[dict[str, Any]]:
    """Table III excerpt — AudioCaps in-domain text→audio."""
    return [
        {"method": "Original", "R1": 28.36, "R5": 61.13, "R10": 75.59, "mAP10": 42.25},
        {"method": "PLSHead", "R1": 28.97, "R5": 62.99, "R10": 76.66, "mAP10": 43.13},
        {"method": "PLSHeadW", "R1": 29.36, "R5": 63.39, "R10": 76.97, "mAP10": 43.49},
    ]


def table_iv_projection_decoding() -> dict[str, float]:
    """Table IV — PD characterization metrics."""
    cfg = CometConfig()
    return {
        "simcos_mean_before": cfg.pd_mean_cos_before,
        "simcos_mean_after": cfg.pd_mean_cos_after,
        "simcos_head": cfg.pd_head_cos,
        "simcos_tail": cfg.pd_tail_cos,
    }


def table_v_captioning_clotho() -> list[dict[str, Any]]:
    """Table V excerpt — Clotho, HTSAT-BERT-ZS."""
    return [
        {"method": "t→a AD", "SPIDEr": 17.6},
        {"method": "t→a PD", "SPIDEr": 27.7},
        {"method": "t100→a100", "SPIDEr": 27.5},
        {"method": "a→a", "SPIDEr": 26.8},
        {"method": "t-924→a-924", "SPIDEr": 6.9},
    ]


def table_v_captioning_audiocaps() -> list[dict[str, Any]]:
    """Table V excerpt — AudioCaps, HTSAT-BERT-ZS."""
    return [
        {"method": "t→a AD", "SPIDEr": 23.3},
        {"method": "t→a PD", "SPIDEr": 41.5},
        {"method": "t100→a100", "SPIDEr": 40.6},
        {"method": "a→a", "SPIDEr": 40.3},
    ]


def headline_results() -> dict[str, Any]:
    return {
        "head_size": "~100 shared semantic axes (comet-like latent structure)",
        "retrieval": "PLSHead matches/beats 1024-D on Clotho & AudioCaps (Table III)",
        "captioning": "t100→a100 ≈ PD and near fully-supervised a→a (Table V)",
        "pd_explanation": "Linear PD = mean shift + head filter (U^T V) + tail rescaling (X̂^T X̂)",
        "compression": "90% dimension reduction with 202 stored vectors",
    }


def pipeline_demo(cfg: CometConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    """Toy PLS-SVD on synthetic paired embeddings."""
    cfg = cfg or CometConfig()
    rng = np.random.default_rng(seed)
    n, c = 64, cfg.embed_dim
    # Shared low-rank structure + modality offsets
    shared = rng.standard_normal((n, cfg.head_size))
    u_shared = rng.standard_normal((c, cfg.head_size))
    v_shared = u_shared + 0.05 * rng.standard_normal((c, cfg.head_size))
    text = shared @ u_shared.T + 0.1 * rng.standard_normal((n, c)) + 0.2
    audio = shared @ v_shared.T + 0.1 * rng.standard_normal((n, c)) - 0.15

    decomp = pls_svd(text, audio)
    sigma = decomp["sigma"]
    uv = np.array([decomp["U"][:, j] @ decomp["V"][:, j] for j in range(min(10, c))])
    t_hat = project_coefficients(decomp["T"], decomp["U"])
    a_hat = project_coefficients(decomp["A"], decomp["V"])
    idx = 0
    cov0 = covariance_decomposition(t_hat[:, idx], a_hat[:, idx])
    net0 = net_useful_contribution(float(sigma[idx]), float(uv[idx]))
    sim = similarity_direct_cross(t_hat[0], a_hat[0], decomp["U"], decomp["V"], head=cfg.head_size)
    head_t = head_energy_ratio(t_hat[0], head_size=cfg.head_size)

    # PLSHead truncate + ES on one sample
    uv_align = uv
    t100 = plshead_truncate(text[0], mean=decomp["t_mean"], directions=decomp["U"], head_size=cfg.head_size)
    a100 = plshead_truncate(audio[0], mean=decomp["a_mean"], directions=decomp["V"], head_size=cfg.head_size)
    a_es = embedding_shift(audio[0], text_mean=decomp["t_mean"], audio_mean=decomp["a_mean"])
    pd_lin = linear_projection_decoding(audio[0], text, u=decomp["U"], v=decomp["V"])
    retrieval = retrieval_smoke(cfg, seed=seed)

    return {
        "num_pairs": n,
        "embed_dim": c,
        "head_size": cfg.head_size,
        "sigma_top3": [round(float(s), 4) for s in sigma[:3]],
        "sigma_tail_mean": round(float(np.mean(sigma[cfg.head_size : cfg.head_size + 20])), 6),
        "uv_align_top3": [round(float(x), 4) for x in uv[:3]],
        "cov_axis0": {k: round(v, 4) if isinstance(v, float) else v for k, v in cov0.items()},
        "net_useful_axis0": round(net0, 4),
        "similarity_direct": round(sim["direct"], 4),
        "similarity_cross": round(sim["cross"], 4),
        "head_energy_text": {k: round(v, 4) for k, v in head_t.items()},
        "plshead_dims": {"text": int(t100.size), "audio": int(a100.size)},
        "embedding_shift_cos_to_text_mean": round(cosine_sim(a_es, decomp["t_mean"]), 4),
        "linear_pd_cos_to_text_mean": round(cosine_sim(pd_lin, decomp["t_mean"]), 4),
        "retrieval_R1": retrieval["R1"],
        "retrieval_R5": retrieval["R5"],
    }


def evaluation_demo(*, seed: int = 0) -> dict[str, Any]:
    demo = pipeline_demo(seed=seed)
    return {
        "headline": headline_results(),
        "demo": demo,
        "framework": framework_card(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    cfg = CometConfig()
    return {
        "table_i_contribution": table_i_contribution(),
        "table_iii_retrieval_clotho": table_iii_retrieval_clotho(),
        "table_iii_retrieval_audiocaps": table_iii_retrieval_audiocaps(),
        "table_iv_projection_decoding": table_iv_projection_decoding(),
        "table_v_captioning_clotho": table_v_captioning_clotho(),
        "table_v_captioning_audiocaps": table_v_captioning_audiocaps(),
        "norm_energy_anchors": {
            "norm_head_text": cfg.norm_head_text,
            "norm_tail_text": cfg.norm_tail_text,
            "norm_head_audio": cfg.norm_head_audio,
            "norm_tail_audio": cfg.norm_tail_audio,
        },
        "headline": headline_results(),
        "limitations": list(LIMITATIONS),
    }
