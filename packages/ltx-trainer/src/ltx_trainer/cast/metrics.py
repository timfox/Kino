"""Paper metrics and benchmark tables (Sec. 5–6)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cast.config import (
    KL_AVG_RANK,
    MEAN_RANK,
    NUM_BENCHMARK_SECTIONS,
    ROLLOUT_JSD_AVG_RANK,
    TOP1_KL_WINS,
    TOP1_RO_WINS,
)


def table1_benchmarks() -> list[dict[str, Any]]:
    """Table 1: benchmark suite summary."""
    return [
        {"section": "BioTIME", "type": "ecological", "n_seq": 646, "D": 128, "ordered": False},
        {"section": "Ember Monthly", "type": "energy", "n_seq": 87, "D": 9, "ordered": False},
        {"section": "OWID Dietary", "type": "diet", "n_seq": 135, "D": 25, "ordered": False},
        {"section": "CDC Weekly Deaths", "type": "mortality", "n_seq": 50, "D": 6, "ordered": False},
        {"section": "BLS QCEW", "type": "employment", "n_seq": 53, "D": 13, "ordered": False},
        {"section": "EPA AirData AQI", "type": "air quality", "n_seq": 800, "D": 6, "ordered": True},
        {"section": "NOAA Storm Events", "type": "weather", "n_seq": 49, "D": 32, "ordered": False},
        {"section": "NYC TLC Trip", "type": "mobility", "n_seq": 24, "D": 266, "ordered": False},
        {"section": "Queue Homogeneous", "type": "G/G/1", "n_seq": 10000, "D": 596, "ordered": True},
        {"section": "Queue Nonhomogeneous", "type": "Gt/G/1", "n_seq": 9900, "D": 562, "ordered": True},
        {"section": "Queue Combined", "type": "pooled queues", "n_seq": 19900, "D": 596, "ordered": True},
    ]


def table2_average_ranks() -> list[dict[str, Any]]:
    """Table 2: average ranks across 11 sections."""
    return [
        {"method": "CAST", "kl_avg": KL_AVG_RANK, "ro_avg": ROLLOUT_JSD_AVG_RANK, "mean": MEAN_RANK},
        {"method": "Comp. ETS", "kl_avg": 5.27, "ro_avg": 3.91, "mean": 4.59},
        {"method": "Persistence", "kl_avg": 5.64, "ro_avg": 3.73, "mean": 4.68},
        {"method": "N-HiTS", "kl_avg": 5.27, "ro_avg": 5.55, "mean": 5.41},
        {"method": "Informer", "kl_avg": 6.45, "ro_avg": 7.18, "mean": 6.82},
        {"method": "Transformer", "kl_avg": 9.00, "ro_avg": 6.64, "mean": 7.82},
        {"method": "ilr-VAR", "kl_avg": 7.45, "ro_avg": 8.64, "mean": 8.05},
        {"method": "GRU", "kl_avg": 8.55, "ro_avg": 8.27, "mean": 8.41},
        {"method": "iTransformer", "kl_avg": 8.73, "ro_avg": 9.73, "mean": 9.23},
        {"method": "TCN", "kl_avg": 8.64, "ro_avg": 11.27, "mean": 9.95},
        {"method": "TimeMixer", "kl_avg": 9.00, "ro_avg": 11.36, "mean": 10.18},
        {"method": "TiDE", "kl_avg": 10.82, "ro_avg": 10.18, "mean": 10.50},
        {"method": "LSTM", "kl_avg": 12.27, "ro_avg": 9.45, "mean": 10.86},
        {"method": "Analog succ.", "kl_avg": 12.00, "ro_avg": 11.18, "mean": 11.59},
        {"method": "Autoformer", "kl_avg": 12.55, "ro_avg": 13.64, "mean": 13.09},
        {"method": "DLinear", "kl_avg": 13.09, "ro_avg": 13.36, "mean": 13.23},
    ]


def table3_cast_wins() -> dict[str, Any]:
    return {
        "offline_kl_top1": f"{TOP1_KL_WINS}/{NUM_BENCHMARK_SECTIONS}",
        "rollout_jsd_top1": f"{TOP1_RO_WINS}/{NUM_BENCHMARK_SECTIONS}",
        "offline_kl_top2": f"{NUM_BENCHMARK_SECTIONS}/{NUM_BENCHMARK_SECTIONS}",
    }


def table5_aliasing_experiment() -> list[dict[str, Any]]:
    """Table 5: synthetic latent-kernel aliasing."""
    return [
        {"method": "Fixed-summary optimum", "kl": 0.044893, "jsd": 0.011562, "l1": 0.238730},
        {"method": "Trained current-only neural", "kl": 0.044899, "jsd": 0.011564, "l1": 0.238774},
        {"method": "Phase-aware anchor only", "kl": 0.047840, "jsd": 0.011722, "l1": 0.246619},
        {"method": "CAST oracle local transport", "kl": 0.0, "jsd": 0.0, "l1": 0.0},
        {"method": "Trained CAST", "kl": 1.7e-8, "jsd": 0.0, "l1": 2.3e-4},
    ]


def table4_ablation() -> list[dict[str, Any]]:
    return [
        {"variant": "CAST Full", "kl_wins": "8/11", "ro_wins": "8/11"},
        {"variant": "w/o structural reg.", "kl_wins": "8/11", "ro_wins": "6/11"},
        {"variant": "anchor only", "kl_wins": "7/11", "ro_wins": "7/11"},
        {"variant": "single-head retrieval", "kl_wins": "7/11", "ro_wins": "7/11"},
        {"variant": "fixed local kernel", "kl_wins": "8/11", "ro_wins": "6/11"},
        {"variant": "w/o persistence mix", "kl_wins": "5/11", "ro_wins": "7/11"},
    ]
