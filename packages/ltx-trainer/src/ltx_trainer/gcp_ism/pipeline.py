"""Framework card, figures, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.gcp_ism.config import GCPIsmConfig
from ltx_trainer.gcp_ism.convolution import build_square_kernel, geometric_convolve_1d
from ltx_trainer.gcp_ism.gcp import lattice_count_1d, lattice_count_direct
from ltx_trainer.gcp_ism.rir import inverse_rir_stub


def framework_card(cfg: GCPIsmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GCPIsmConfig()
    return {
        "paper": cfg.paper_arxiv,
        "venue": cfg.venue,
        "title": cfg.title,
        "github": cfg.github,
        "complexity": {
            "direct_ism": cfg.direct_ism_big_o,
            "gcp_ism": cfg.gcp_ism_big_o,
        },
        "constructions": ["forward_finite_difference (Eq. 18)", "inverse_lanczos (Eq. 20)", "TF assembly (Eq. 21)"],
        "constraints": "integer Z^N coordinates; omni source/receiver; angle-independent reflections",
        "demo_room": {
            "s": list(cfg.source),
            "r": list(cfg.receiver),
            "ell": list(cfg.orthotope),
            "T_s": cfg.duration_s,
        },
    }


def fig5_demo_params() -> dict[str, Any]:
    """Figure 5 RIR parameters."""
    return {
        "s": [1, 0, 1],
        "r": [2, 1, 1],
        "ell": [5, 4, 3],
        "Gamma_plus": [0.93, 0.8, 0.9],
        "Gamma_minus": [0.72, 0.78, 0.8],
        "lambda": 1,
        "T": 0.3,
        "lanczos_alpha": 10,
    }


def fig7_high_dim_params() -> dict[str, Any]:
    """Figure 7 — N=4..6 echo density study."""
    return {
        "s": [1, 0, 1, 0, 1, 3],
        "r": [2, 1, 1, 3, 2, 2],
        "ell": [5, 4, 3, 7, 6, 8],
        "Gamma_plus": [0.93, 0.8, 0.9, 0.93, 0.77, 0.82],
        "Gamma_minus": [0.72, 0.78, 0.93, 0.67, 0.52, 0.7],
        "T": 0.3,
        "N_dims": [4, 5, 6],
        "eta_tail_note": "converges ~1.5 with in-phase coeffs; ~1 with phase-inverted",
    }


def fig9_lambda_tradeoff() -> dict[str, Any]:
    """Figure 9 — λ scaling vs NMSE / runtime."""
    return {
        "nmse_drop_per_lambda_doubling_db": 12.0,
        "runtime_growth": "more than 4× per λ doubling",
        "compute": "O(N (kλ)^2 log(kλ))",
    }


def fig11_long_rir() -> dict[str, Any]:
    """Figure 11 — N=6, T60~4s broadband."""
    return {
        "N": 6,
        "T60_target_s": 4.0,
        "T60_xi": 4,
        "metrics": ["EDC Schroeder", "RT60 −10..−30 dB regression", "P50", "echo density η"],
    }


def gcp_count_smoke(k: int, n: int) -> dict[str, int]:
    """Compare 1D / direct recurrence counts at small k."""
    return {
        "k": k,
        "N": n,
        "C_1d": lattice_count_1d(float(k)),
        "C_direct": lattice_count_direct(float(k), n),
    }


def forward_smoke(cfg: GCPIsmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GCPIsmConfig()
    k2 = cfg.demo_k_max * cfg.demo_k_max
    f = build_square_kernel(k2)
    c1 = torch.ones(k2 + 1)
    c2 = geometric_convolve_1d(f, c1)
    rir = inverse_rir_stub(32, alpha=cfg.lanczos_alpha)
    counts = gcp_count_smoke(min(8, cfg.demo_k_max), cfg.demo_n_dims)
    return {
        "conv_shape": list(c2.shape),
        "rir_len": len(rir),
        "rir_energy": float(rir.pow(2).sum().detach()),
        "gcp_counts": counts,
        "lambda_nmse_db": cfg.lambda_nmse_drop_db,
    }


def evaluation_demo(cfg: GCPIsmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GCPIsmConfig()
    return {
        "framework": framework_card(cfg),
        "fig5": fig5_demo_params(),
        "fig7": fig7_high_dim_params(),
        "fig9": fig9_lambda_tradeoff(),
        "fig11": fig11_long_rir(),
        "forward": forward_smoke(cfg),
    }
