"""TSB framework card and paper result excerpts (arXiv:2605.24825)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tsb.config import TsbConfig
from ltx_trainer.tsb.layout import LIMITATIONS
from ltx_trainer.tsb.mock import evaluation_smoke


def framework_card(cfg: TsbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TsbConfig()
    return {
        "name": "Time Segmented Beamforming via Dynamic Programming",
        "paper": cfg.paper_arxiv,
        "companion": cfg.companion_online,
        "authors": "Mittal, Corey, Cuji, Buck, Singer",
        "problem": "Non-stationary interferers smear SCM in fixed-window Capon beamforming",
        "batch_method": "BSB — globally optimal piecewise MVDR via Bellman DP (Eq. 25)",
        "online_method": "OSB — causal OSRLS-style greedy segmentation (Eq. 26)",
        "baseline": "Sample-matrix-inversion Capon / sliding-window MPDR",
        "constraint": "Distortionless: w^T ν = 1",
        "penalty_c": cfg.default_penalty_c,
        "min_segment_tau": cfg.min_segment_length_tau,
        "datasets": ["SwellEx-96 HLA (49 Hz)", "Massive Distributed Microphone Array"],
        "limitations": LIMITATIONS,
    }


def table_simulation_abrupt_change() -> list[dict[str, Any]]:
    """§VII-A — abrupt interference, ULA M=9."""
    return [
        {"method": "Batch Capon (SMI)", "mse_relative_db": 0.0},
        {"method": "Batch Segmented (BSB)", "mse_relative_db": -3.0},
        {"method": "Genie SLS (upper bound)", "mse_relative_db": -3.5},
    ]


def table_piecewise_bearing() -> dict[str, Any]:
    """§VII-C — piecewise constant bearing, T=20000."""
    return {
        "ula_elements": 15,
        "target_snr_db": -5,
        "inr_db": 11,
        "block_length_snapshots": "470–530",
        "finding": "OSB matches best hindsight sliding window (32–1024)",
    }


def table_birth_death() -> dict[str, Any]:
    """§VII-E — Markov birth-death interferers."""
    return {
        "p_birth": 0.02,
        "p_death": 0.001,
        "penalty_c": 4.8,
        "min_segment_tau": 5,
        "finding": "OSB resets covariance at jumps; cumulative MSE ≤ best fixed window",
    }


def table_swellex96() -> dict[str, Any]:
    """§VIII-A — SwellEx-96 HLA at 49 Hz."""
    return {
        "elements": 28,
        "segment_minutes": 10,
        "steering_grid_deg": "−90 to 90 step 1",
        "osb_penalty_c": 0.1,
        "osb_tau": 1,
        "osb_search_k": 60,
        "source_bearing_deg": 43,
        "finding": "Accumulated power at 43° on par with best sliding MPDR",
    }


def table_distributed_mic() -> dict[str, Any]:
    """§VIII-B — 40-channel cocktail party."""
    return {
        "channels": 40,
        "room_t60_ms": 800,
        "stft_frame": 1024,
        "talkers": 4,
        "metrics": ["SI-SDR", "PESQ"],
        "finding": "OSB beats fixed-window MPDR on reverberant speech",
    }


def theorem_regret() -> dict[str, Any]:
    """Theorem 1 — Eq. (32)."""
    return {
        "statement": (
            "L_alg(T) − L_batch(P*) ≤ K* (max(C, τ L_max) + (d/2) ln(T/K*) + Γ)"
        ),
        "growth": "O(K* log T) — logarithmic in horizon per segment count",
    }


def headline_results() -> dict[str, Any]:
    abrupt = table_simulation_abrupt_change()
    return {
        "finding": (
            "Segmented MVDR partitions non-stationary snapshots by local stationarity, "
            "avoiding SCM smearing. BSB ~3 dB MSE gain vs batch Capon in abrupt-change sims; "
            "OSB matches hindsight-optimal sliding windows on piecewise and birth-death scenes."
        ),
        "bsb_mse_gain_db": abs(abrupt[1]["mse_relative_db"]),
        "regret": theorem_regret()["growth"],
        "swellex_matches_best_sliding": True,
    }


def evaluation_demo(cfg: TsbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TsbConfig()
    return {"paper": cfg.paper_arxiv, "smoke": evaluation_smoke()}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "simulation_abrupt_change": table_simulation_abrupt_change(),
        "piecewise_bearing": table_piecewise_bearing(),
        "birth_death": table_birth_death(),
        "swellex96": table_swellex96(),
        "distributed_mic": table_distributed_mic(),
        "theorem_regret": theorem_regret(),
        "headlines": headline_results(),
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
