"""Literature anchors from §4 (Fig. 6–9)."""

from __future__ import annotations

from typing import Any


def operating_guidelines() -> list[str]:
    return [
        "Deploy BBR-Copilot beside unmodified BBR on QUIC CDN paths (Layer-2 copilot).",
        "Generate padding only when application-limited AND pacing_gain > 1.",
        "Receiver ACKs padding but does not deliver filler bytes to the application.",
        "Lost padding is not retransmitted — avoids blocking live frame recovery.",
        "High RTT live streams suffer more inaccurate BBR samples without copilot.",
        "Balance extra probe bytes vs bitrate gap in future work (§5).",
    ]


def table_startup_fig7() -> dict[str, Any]:
    return {
        "bitrate_mbps": 5.4,
        "buffer_kb": 40,
        "rtt_ms_sweep": [50, 100, 150, 200],
        "baseline_exit_pct_at_200ms": 3.0,
        "copilot_exit_pct_all_rtt": 100.0,
        "max_retrans_optimization_pct_at_200ms": 43.3,
        "baseline_retrans_pct_200ms": 24.8,
        "copilot_retrans_pct_200ms": 14.1,
    }


def table_probe_fig8_50ms() -> dict[str, Any]:
    return {
        "rtt_ms": 50,
        "uplink_step_mbps": "12→24→12",
        "baseline_rmse_mbps_window_20_40s": 9.41,
        "copilot_rmse_mbps_window_20_40s": 1.31,
        "rmse_reduction_pct": 86.1,
        "baseline_peak_sample_mbps": 16.4,
        "true_peak_mbps": 24.0,
    }


def table_probe_fig9_200ms() -> dict[str, Any]:
    return {
        "rtt_ms": 200,
        "baseline_rmse_mbps_window_20_40s": 12.18,
        "copilot_rmse_mbps_window_20_40s": 2.72,
        "rmse_reduction_pct": 77.7,
        "copilot_peak_sample_mbps": 23.0,
    }


def table_production_stats() -> dict[str, Any]:
    return {
        "live_flows_measured": 570_000,
        "bbr_stuck_startup_pct": 88.4,
        "quic_loc_lines": 200,
        "industrial_adopters": ["Amazon", "Tencent", "ByteDance", "Huawei"],
    }
