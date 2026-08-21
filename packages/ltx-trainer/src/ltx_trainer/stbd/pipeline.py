"""Subspace TBD framework card and paper tables (arXiv:2605.25498)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.stbd.config import StbdConfig
from ltx_trainer.stbd.layout import LIMITATIONS
from ltx_trainer.stbd.mock import compare_likelihoods_at_true_state, ring_microphones


def framework_card(cfg: StbdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StbdConfig()
    return {
        "name": "Subspace TBD",
        "paper": cfg.paper_arxiv,
        "venue": cfg.venue,
        "authors": "Nobutaka Ito, Yoshiaki Bando (AIST)",
        "idea": (
            "Passive multi-target track-before-detect using complex Bingham likelihood "
            "on normalized multichannel STFT observations. Each hypothesis maps to a "
            "steering-vector subspace; unknown emitted signals are not estimated."
        ),
        "observation": "z_tf = ez_tf / ||ez_tf||_2; likelihood ∝ exp(Σ_f κ_f ||P_f z_tf||²)",
        "baseline": "Deterministic-contribution superpositional TBD [8] (coefficients = 1)",
        "inference": "Auxiliary particle filter; given activity pattern a_t",
        "simulation": {
            "room_m": cfg.room_size_m,
            "microphones": cfg.num_sensors,
            "frames": cfg.num_frames,
            "target_slots": cfg.num_target_slots,
            "freq_bins": cfg.num_freq_bins,
        },
        "defaults": cfg.__dict__,
    }


def table_i_median_rmse() -> list[dict[str, Any]]:
    """Table I — median position RMSE (m) over five runs."""
    return [
        {"snr_db": -10, "np": 2000, "conventional": 1.0098, "proposed": 0.0305, "proposed_range": (0.0249, 0.0929)},
        {"snr_db": -10, "np": 4000, "conventional": 0.9786, "proposed": 0.0248, "proposed_range": (0.0225, 0.0615)},
        {"snr_db": -10, "np": 8000, "conventional": 1.0930, "proposed": 0.0239, "proposed_range": (0.0230, 0.0323)},
        {"snr_db": 0, "np": 2000, "conventional": 1.0780, "proposed": 0.0230, "proposed_range": (0.0181, 0.6544)},
        {"snr_db": 0, "np": 4000, "conventional": 0.9411, "proposed": 0.0234, "proposed_range": (0.0139, 0.0263)},
        {"snr_db": 0, "np": 8000, "conventional": 0.8538, "proposed": 0.0113, "proposed_range": (0.0092, 0.0134)},
        {"snr_db": 10, "np": 2000, "conventional": 1.0503, "proposed": 0.0169, "proposed_range": (0.0145, 0.0435)},
        {"snr_db": 10, "np": 4000, "conventional": 1.0070, "proposed": 0.0264, "proposed_range": (0.0074, 0.0332)},
        {"snr_db": 10, "np": 8000, "conventional": 1.0294, "proposed": 0.0074, "proposed_range": (0.0062, 0.0158)},
    ]


def trajectory_example_rmse() -> dict[str, float]:
    """Figs. 2–3 single-run RMSE at −10 dB, np=2000."""
    return {"proposed_m": 0.0325, "conventional_m": 0.9363, "snr_db": -10.0, "np": 2000}


def headline_results() -> dict[str, Any]:
    t = table_i_median_rmse()
    minus10 = [r for r in t if r["snr_db"] == -10]
    return {
        "finding": (
            "At −10 dB SNR, proposed subspace TBD median RMSE < 0.031 m vs "
            "conventional ~1 m; tracks two targets with unknown emitted signals."
        ),
        "trajectory_example": trajectory_example_rmse(),
        "median_rmse_minus10_db": {r["np"]: r["proposed"] for r in minus10},
        "conventional_median_minus10_db": {r["np"]: r["conventional"] for r in minus10},
        "kappa": 10.0,
        "activity_schedule": "1 target frames 0–99; 2 targets from frame 100",
    }


def evaluation_demo(cfg: StbdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or StbdConfig()
    import numpy as np

    rng = np.random.default_rng(99)
    mic = ring_microphones(cfg.num_sensors, cfg.room_size_m)
    pos = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
    cmp = compare_likelihoods_at_true_state(mic, pos, frequency_hz=200.0, rng=rng, snr_db=-10.0)
    return {
        "num_microphones": mic.shape[0],
        "likelihood_comparison": cmp,
        "kappa": cfg.kappa_per_freq,
        "trajectory_rmse_example": trajectory_example_rmse(),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_median_rmse": table_i_median_rmse(),
        "trajectory_example_rmse": trajectory_example_rmse(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
