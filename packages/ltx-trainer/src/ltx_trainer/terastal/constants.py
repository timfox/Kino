"""Terastal: layer-variant scheduling for heterogeneous DNN accelerators (arXiv:2606.06818)."""

from __future__ import annotations

PAPER_ARXIV = "2606.06818"
PAPER_TITLE = (
    "Terastal: Layer-Variant-based Scheduling for Real-Time Multi-DNN "
    "Workloads on Heterogeneous Accelerators"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_VENUE = "RTCSA 2026"

DATAFLOWS = ("ws", "os")  # weight-stationary, output-stationary

# Paper anchor reductions in deadline miss rate (per-model average)
MISS_RATE_REDUCTION_VS = {
    "fcfs": 0.4058,
    "edf": 0.3053,
    "dream": 0.3627,
}
AVG_NORMALIZED_ACCURACY_LOSS = 0.0224
DEFAULT_ACCURACY_THRESHOLD = 0.90  # 90% of baseline

# Hardware settings (Table I)
HARDWARE_SETTINGS = (
    {"pes": 4000, "partition": "1_ws_2k_2_os_1k", "scenarios": ("ar_social", "ar_gaming_light", "mcv_light")},
    {"pes": 6000, "partition": "1_ws_2k_2_os_2k", "scenarios": ("ar_social", "ar_gaming_heavy", "mcv_heavy")},
)

# Simulator stack cited in paper
SIMULATOR_STACK = ("maestro", "xrbench")
