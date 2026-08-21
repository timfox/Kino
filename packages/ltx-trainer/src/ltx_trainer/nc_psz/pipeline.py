"""Framework card and paper benchmark excerpts (arXiv:2605.21891)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nc_psz.config import NcPszConfig
from ltx_trainer.nc_psz.layout import LIMITATIONS
from ltx_trainer.nc_psz.mock import evaluation_smoke


def framework_card(cfg: NcPszConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NcPszConfig()
    return {
        "name": "Neighbor-consistent neural filters for PSZ under localization uncertainty",
        "paper": cfg.paper_arxiv,
        "authors": "Hao Jiang, Edgar Choueiri (Princeton 3D3A)",
        "system": {
            "split_band": "woofer 100–2000 Hz + tweeter 2–20 kHz",
            "generators": "f_θw(x), f_θt(x) Fourier-encoded MLPs",
            "baseline_loss": "BSANN-style L_psz (bright/dark + gain + compact)",
            "regularization": "L_nc penalizes ||g(x)−g(x′)||² for perturbed coordinates x′",
        },
        "evaluation": {
            "decoupled_protocol": "ATFs fixed at physical x; filters from perturbed x̂ only",
            "quality_metrics": "IZI, IPI → median + CVaR10 (sim) or min (meas)",
            "stability_metrics": "σ_mean, σ_rms variation rate (dB/m)",
        },
        "hyperparameters": {
            "delta_m": cfg.nc_delta_m,
            "lambda_w": cfg.nc_lambda_woofer,
            "lambda_t": cfg.nc_lambda_tweeter,
        },
        "headlines": headline_results(),
        "limitations": LIMITATIONS,
    }


def table_i_simulation_woofer() -> list[dict[str, Any]]:
    """Table I: simulation woofer anchor-averaged summaries."""
    return [
        {
            "metric": "IZI",
            "median_base": 9.35,
            "median_nc": 9.41,
            "cvar10_base": 8.71,
            "cvar10_nc": 8.76,
            "sigma_mean_base": 5.99,
            "sigma_mean_nc": 4.85,
            "sigma_rms_base": 9.07,
            "sigma_rms_nc": 6.63,
        },
        {
            "metric": "IPI",
            "median_base": 9.57,
            "median_nc": 9.54,
            "cvar10_base": 8.77,
            "cvar10_nc": 9.07,
            "sigma_mean_base": 6.83,
            "sigma_mean_nc": 3.36,
            "sigma_rms_base": 9.92,
            "sigma_rms_nc": 4.38,
        },
    ]


def table_ii_simulation_tweeter() -> list[dict[str, Any]]:
    """Table II: simulation tweeter anchor-averaged summaries."""
    return [
        {
            "metric": "IZI",
            "median_base": 10.42,
            "median_nc": 10.78,
            "cvar10_base": 9.36,
            "cvar10_nc": 9.89,
            "sigma_mean_base": 9.35,
            "sigma_mean_nc": 6.65,
            "sigma_rms_base": 11.89,
            "sigma_rms_nc": 8.28,
        },
        {
            "metric": "IPI",
            "median_base": 10.90,
            "median_nc": 11.02,
            "cvar10_base": 9.30,
            "cvar10_nc": 9.51,
            "sigma_mean_base": 14.73,
            "sigma_mean_nc": 11.07,
            "sigma_rms_base": 19.32,
            "sigma_rms_nc": 14.68,
        },
    ]


def table_iii_measurements_excerpt() -> list[dict[str, Any]]:
    """Table III excerpt: L2 IZI min and L1 σ_mean at 5 cm spacing."""
    return [
        {
            "spacing_m": 0.05,
            "listener": "L2",
            "metric": "IZI min",
            "baseline": 6.10,
            "nc": 6.83,
            "imp_pct": 11.9,
        },
        {
            "spacing_m": 0.05,
            "listener": "L1",
            "metric": "IZI σ_mean",
            "baseline": 5.08,
            "nc": 3.14,
            "imp_pct": 38.2,
        },
        {
            "spacing_m": 0.02,
            "listener": "L1",
            "metric": "IZI σ_mean",
            "baseline": 4.61,
            "nc": 1.76,
            "imp_pct": 61.8,
        },
        {
            "spacing_m": 0.10,
            "listener": "L2",
            "metric": "IZI min",
            "baseline": 5.36,
            "nc": 6.27,
            "imp_pct": 16.9,
        },
    ]


def headline_results() -> dict[str, Any]:
    cfg = NcPszConfig()
    return {
        "woofer_LPH_IPI_sigma_rms_improvement_pct": cfg.woofer_ipi_sigma_rms_imp_pct,
        "tweeter_LPH_IZI_sigma_rms_improvement_pct": cfg.tweeter_izi_sigma_rms_imp_pct,
        "measurement_L2_IZI_min_max_imp_pct": cfg.meas_l2_izi_min_imp_pct,
        "measurement_L1_IZI_sigma_mean_max_imp_pct": cfg.meas_l1_izi_sigma_mean_imp_pct,
        "nc_delta_m": cfg.nc_delta_m,
        "nc_lambda": cfg.nc_lambda_woofer,
    }


def evaluation_demo(cfg: NcPszConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NcPszConfig()
    return {"config": cfg.__dict__, "smoke": evaluation_smoke(cfg)}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_simulation_woofer": table_i_simulation_woofer(),
        "table_ii_simulation_tweeter": table_ii_simulation_tweeter(),
        "table_iii_measurements_excerpt": table_iii_measurements_excerpt(),
        "headlines": headline_results(),
    }
