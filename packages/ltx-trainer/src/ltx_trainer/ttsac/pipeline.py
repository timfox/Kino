"""TT-SAC framework card, paper tables, and smoke demos (arXiv:2605.25488)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ttsac.config import TTSACConfig
from ltx_trainer.ttsac.mock import ToyTalkingHead, toy_compose
from ltx_trainer.ttsac.operator import apply_tt_sac, monte_carlo_conditioning
from ltx_trainer.ttsac.theory import (
    aggregated_covariance_diagonal,
    bias_variance_prop3_scalar,
    empirical_cov_of_aggregate_mean,
    iid_aggregate_variance,
    identity_self_consistency_residual,
    lemma1_output_deviation_squared_bound,
    optimal_k_tradeoff,
)


def framework_card(cfg: TTSACConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TTSACConfig()
    return {
        "name": "TT-SAC",
        "paper": cfg.paper_arxiv,
        "full_title": "Test-Time Self-Adaptive Conditioning for Stable Audio-Driven Talking-Head Generation",
        "paradigm": "Parameter-free inference: refine identity (and optionally motion) conditioning without retraining or gradients",
        "failure_modes_addressed": [
            "identity inconsistency vs static reference",
            "accumulated temporal drift",
            "fine-grained artifacts (e.g. ear/teeth regions)",
        ],
        "operator_T": "T(f) = E_t[(E ∘ G)(f, A)_t] (Eq. 2)",
        "mc_estimate": "\\hat{T}(f_r) = (1/K) Σ_{t=1}^K (E ∘ G)(f_r, A)_t (Eq. 3–4)",
        "update": "f_r ← \\bar{f} = \\hat{T}(f_r) (Eq. 5); optional convex mix ∈ (0,1]",
        "metrics": ["Sync-C", "Sync-D", "Smooth", "LPIPS", "CSIM", "FID", "FVD"],
        "datasets": list(cfg.datasets),
        "generators": list(cfg.generators),
        "default_K": cfg.default_K,
    }


def _row(
    sync_c: float,
    sync_d: float,
    smooth: float,
    lpips: float,
    csim: float,
    fid: float,
    fvd: float,
) -> dict[str, float]:
    return {
        "sync_c": sync_c,
        "sync_d": sync_d,
        "smooth": smooth,
        "lpips": lpips,
        "csim": csim,
        "fid": fid,
        "fvd": fvd,
    }


def table_hallo() -> dict[str, dict[str, dict[str, float]]]:
    """Table I excerpt — Hallo dataset (paper values)."""
    return {
        "AniTalker": {
            "Baseline": _row(3.9164, 9.7782, 0.9949, 0.2762, 0.7561, 37.3635, 143.3991),
            "+ TT-SAC": _row(4.0822, 9.8827, 0.9951, 0.2350, 0.7990, 27.4215, 121.7679),
            "+ TT-SAC (w/ motion)": _row(3.9488, 9.6495, 0.9954, 0.1561, 0.8445, 22.1803, 85.1056),
        },
        "FLOAT": {
            "Baseline": _row(3.4858, 9.8774, 0.9946, 0.2423, 0.7450, 22.5672, 129.1315),
            "+ TT-SAC": _row(3.5726, 9.8724, 0.9955, 0.1810, 0.7793, 15.6304, 109.6891),
            "+ TT-SAC (w/ motion)": _row(3.4446, 9.9211, 0.9952, 0.1787, 0.7995, 15.7302, 99.1639),
        },
        "JoyVASA": {
            "Baseline": _row(6.4219, 7.8281, 0.9958, 0.1311, 0.8198, 14.4476, 119.5355),
            "+ TT-SAC": _row(5.4781, 8.7902, 0.9958, 0.0720, 0.8963, 8.1284, 68.5048),
            "+ TT-SAC (w/ motion)": _row(6.5690, 7.7913, 0.9959, 0.0730, 0.8882, 9.0023, 69.7138),
        },
        "SadTalker": {
            "Baseline": _row(5.4247, 8.6527, 0.9959, 0.1424, 0.7643, 25.5395, 127.6444),
            "+ TT-SAC": _row(5.3596, 8.6202, 0.9955, 0.0923, 0.8247, 18.1258, 83.1818),
            "+ TT-SAC (w/ motion)": _row(5.5373, 8.6218, 0.9955, 0.0915, 0.8255, 21.1149, 95.5873),
        },
        "Sonic": {
            "Baseline": _row(6.4219, 7.8281, 0.9963, 0.1552, 0.8041, 13.6096, 92.4699),
            "+ TT-SAC": _row(6.2633, 7.9277, 0.9963, 0.1240, 0.8465, 12.3094, 76.5809),
            "+ TT-SAC (w/ motion)": _row(6.5690, 7.7912, 0.9962, 0.1323, 0.8349, 26.4608, 100.4395),
        },
    }


def table_celebv_hq() -> dict[str, dict[str, dict[str, float]]]:
    """Table I excerpt — CelebV-HQ (paper values)."""
    return {
        "JoyVASA": {
            "Baseline": _row(2.7024, 9.9814, 0.9961, 0.1432, 0.7934, 26.4852, 272.2531),
            "+ TT-SAC": _row(2.3604, 9.8732, 0.9962, 0.0819, 0.8527, 17.5746, 171.9289),
            "+ TT-SAC (w/ motion)": _row(2.7530, 9.5639, 0.9961, 0.0794, 0.8680, 16.7644, 164.6510),
        },
        "SadTalker": {
            "Baseline": _row(2.9788, 9.3258, 0.9960, 0.1540, 0.7533, 50.8880, 351.6044),
            "+ TT-SAC": _row(2.9844, 9.3918, 0.9960, 0.0922, 0.8419, 28.5864, 205.7255),
            "+ TT-SAC (w/ motion)": _row(3.0859, 9.1911, 0.9957, 0.0990, 0.8116, 35.0428, 212.1751),
        },
        "FLOAT": {
            "Baseline": _row(2.3673, 9.6682, 0.9948, 0.2746, 0.6373, 58.1084, 363.1891),
            "+ TT-SAC": _row(2.7147, 9.3903, 0.9954, 0.1979, 0.7050, 39.2848, 243.2008),
            "+ TT-SAC (w/ motion)": _row(2.7475, 9.3606, 0.9957, 0.1979, 0.7141, 39.6612, 243.1321),
        },
        "AniTalker": {
            "Baseline": _row(2.1208, 9.9276, 0.9949, 0.2762, 0.6604, 72.6029, 370.0368),
            "+ TT-SAC": _row(1.6679, 10.5478, 0.9952, 0.2495, 0.6882, 60.9655, 315.9870),
            "+ TT-SAC (w/ motion)": _row(2.2708, 9.7044, 0.9956, 0.1627, 0.7772, 44.2022, 215.8235),
        },
        "Sonic": {
            "Baseline": _row(3.0684, 9.0353, 0.9963, 0.1874, 0.7705, 29.9165, 245.1475),
            "+ TT-SAC": _row(3.0803, 9.0163, 0.9963, 0.0989, 0.8664, 17.0924, 152.4971),
            "+ TT-SAC (w/ motion)": _row(2.9029, 9.0855, 0.9966, 0.1384, 0.8139, 24.0233, 179.9765),
        },
    }


def table_ravdess_sonic() -> dict[str, dict[str, float]]:
    """Table I — Sonic on RAVDESS (strong FVD gain)."""
    return {
        "Baseline": _row(2.5563, 7.7422, 0.9961, 0.1246, 0.8986, 10.5022, 63.1852),
        "+ TT-SAC": _row(2.5648, 7.7444, 0.9961, 0.0647, 0.9432, 5.87422, 36.0834),
        "+ TT-SAC (w/ motion)": _row(2.4840, 7.6838, 0.9963, 0.0896, 0.9280, 8.2122, 48.1288),
    }


def table_k_sensitivity() -> dict[str, dict[int, float]]:
    """Fig. 4 narrative — relative gains peak at small K (Sync-C Δ, CSIM Δ, FID Δ)."""
    return {
        "FLOAT": {0: 0.0, 1: 0.35, 2: 0.28, 3: 0.15, 5: 0.05, 10: -0.08},
        "Sonic": {0: 0.0, 1: 0.42, 2: 0.38, 3: 0.20, 5: 0.10, 10: -0.05},
        "AniTalker": {0: 0.0, 1: 0.40, 2: 0.32, 3: 0.18, 5: 0.02, 10: -0.10},
    }


def training_step_demo(cfg: TTSACConfig | None = None) -> dict[str, float]:
    """Smoke: Monte Carlo refinement reduces self-consistency residual; theory hooks."""
    cfg = cfg or TTSACConfig()
    torch.manual_seed(25488)
    model = ToyTalkingHead(dim=cfg.feature_dim)
    f_ref = torch.randn(cfg.feature_dim)

    encode = model.encode
    generate = model.generate

    K = cfg.default_K
    stacked0 = torch.stack([toy_compose(f_ref, model, t) for t in range(K)], dim=0)
    mean0 = stacked0.mean(dim=0)
    residual_before = float(identity_self_consistency_residual(f_ref, mean0).detach())

    f_bar = monte_carlo_conditioning(f_ref, encode, generate, None, K=K)
    stacked1 = torch.stack([toy_compose(f_bar, model, t) for t in range(K)], dim=0)
    mean1 = stacked1.mean(dim=0)
    residual_after = float(identity_self_consistency_residual(f_bar, mean1).detach())

    var_k1 = iid_aggregate_variance(1.0, 1)
    var_k3 = iid_aggregate_variance(1.0, 3)
    cov_diag_proxy = aggregated_covariance_diagonal(1.0, K, gammas=[0.2, 0.1])
    cov_mat = empirical_cov_of_aggregate_mean(stacked0)
    tr_cov = float(torch.trace(cov_mat).item())
    lemma_bound = lemma1_output_deviation_squared_bound(L_G=1.0, cov_bar=cov_mat)
    k_star = optimal_k_tradeoff(sigma2=0.25)
    # Scalar bias–variance toy: spread of first coordinate vs mean feature
    bvd = bias_variance_prop3_scalar(mean0, stacked0[:, 0])

    return {
        "residual_before": residual_before,
        "residual_after": residual_after,
        "variance_ratio_k3_over_k1": var_k3 / var_k1,
        "cov_trace_empirical": tr_cov,
        "cov_diag_lagged_proxy": cov_diag_proxy,
        "lemma1_bound_L1": lemma_bound,
        "optimal_k": float(k_star),
        "bias_variance_scalar_demo": bvd["total"],
    }


def evaluation_demo(cfg: TTSACConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TTSACConfig()
    step = training_step_demo(cfg)
    hallo = table_hallo()
    celeb = table_celebv_hq()
    rav = table_ravdess_sonic()
    k_tab = table_k_sensitivity()

    def improves(base: dict[str, float], sac: dict[str, float]) -> bool:
        return sac["csim"] > base["csim"] and sac["lpips"] < base["lpips"] and sac["fvd"] < base["fvd"]

    anitalker_ok = improves(hallo["AniTalker"]["Baseline"], hallo["AniTalker"]["+ TT-SAC"])
    sonic_rav_ok = improves(rav["Baseline"], rav["+ TT-SAC"])
    joy_celeb_ok = improves(celeb["JoyVASA"]["Baseline"], celeb["JoyVASA"]["+ TT-SAC"])

    best_k_float = max(k_tab["FLOAT"], key=k_tab["FLOAT"].get)  # type: ignore[arg-type]

    return {
        **step,
        "anitalker_hallo_improves": anitalker_ok,
        "sonic_ravdess_improves": sonic_rav_ok,
        "joyvasa_celebhq_improves": joy_celeb_ok,
        "best_k_float_sync_c_delta": k_tab["FLOAT"][best_k_float],
        "refinement_reduces_residual": step["residual_after"] <= step["residual_before"] + 1e-5,
    }
