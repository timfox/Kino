"""Paper tables and headline numbers (arXiv:2605.28717)."""

from __future__ import annotations

from typing import Any

TABLE2_RESOURCES: dict[str, Any] = {
    "target": "Xilinx Alveo U50 @ 322 MHz",
    "openurma": {"lut": 122_710, "ff": 194_266, "bram18": 328, "lut_pct": 14.1},
    "openroce": {"lut": 46_636, "ff": 91_900, "bram18": 67, "lut_pct": 5.4},
    "lut_ratio": 2.63,
}

TABLE5_STATE: list[dict[str, Any]] = [
    {"n": 1, "m": 1, "openurma_bytes": 108, "roce_bytes": 544, "ratio": 5.0},
    {"n": 1024, "m": 1024, "openurma_kb": 110.6, "roce_mb": 536.9, "ratio": 4855.0},
]

TABLE6_LATENCY_HEADLINE: dict[str, int] = {
    "UB_LD/ST": 500,
    "UB_URMA": 757,
    "RoCE_BF": 1686,
    "RoCE_DMA": 2186,
}

TABLE8_LOAD_BEARING: list[dict[str, Any]] = [
    {
        "remove": "Bounded state (layer split)",
        "cost": "NIC SRAM spill → per-op context refetch",
        "impact": "+~1000 ns/op at N≳23",
    },
    {
        "remove": "On-bus controller",
        "cost": "4 PCIe traversals + target-side DMA",
        "impact": "4.37× → 1× on 64 B READ",
    },
    {
        "remove": "Opt-in ordering",
        "cost": "Always-on strict PSN order + HOL blocking",
        "impact": "+50 ns every WR",
    },
]

PILOT_STACKS: list[dict[str, Any]] = [
    {"stack": "UB LD/ST", "tx_cycles_cold": 8, "e2e_ns": 500},
    {"stack": "UB URMA", "tx_cycles_cold": 24, "e2e_ns": 757},
    {"stack": "OpenRoCE", "tx_cycles_cold": 9, "e2e_ns": 2186},
]

LOCOMO_4X: dict[str, Any] = {
    "openurma_f1": 41.94,
    "longllmlingua_f1": 43.94,
    "e2e_ms_openurma": 480.6,
    "e2e_ms_longllm": 2006.4,
}
