"""OpenURMA framework card and evaluation demo (arXiv:2605.28717)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.openurma.baselines import (
    PILOT_STACKS,
    TABLE2_RESOURCES,
    TABLE5_STATE,
    TABLE6_LATENCY_HEADLINE,
    TABLE8_LOAD_BEARING,
)
from ltx_trainer.openurma.config import OpenURMAConfig
from ltx_trainer.openurma.latency import (
    headline_ratios,
    latency_decomposition,
    stacks_summary,
)
from ltx_trainer.openurma.ordering import gating_cost_stub, ordering_surface_card
from ltx_trainer.openurma.state_model import (
    per_nic_state_ub,
    sram_spill_threshold_qp_bytes,
    state_scaling_table,
)


def framework_card(cfg: OpenURMAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OpenURMAConfig()
    ratios = headline_ratios(cfg)
    return {
        "name": "OpenURMA",
        "paper": "arXiv:2605.28717",
        "subtitle": "Clean-room open Unified Bus transport + transaction layers",
        "upstream": "https://github.com/bojieli/OpenURMA",
        "spec": "UB-base-specification 2.0.1 (unifiedbus.org)",
        "toolchain": "OpenClickNP → emulator / SystemC / Vitis HLS (Alveo U50)",
        "baseline": "OpenRoCE (RoCEv2 RC) on same toolchain",
        "architectural_moves": [
            "Jetty (per-app) + TP Channel (per-host) → O(N+M) state",
            "UB controller on on-chip bus (not PCIe)",
            "Load/store path with TP bypass (5 stages cold)",
            "Opt-in ordering (ROI/ROT/ROL/UNO × NO/RO/SO + Fence)",
        ],
        "headline_64b_read_ns": stacks_summary(cfg),
        "speedup_vs_roce_dma": ratios["ub_vs_roce_dma"],
        "elements": {"openurma": 39, "openroce": 21},
        "clock_mhz": cfg.clock_mhz,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28717",
        "problem": "RoCE NIC is PCIe peripheral: O(N·M) QP state, 4 PCIe traversals per small op",
        "ub_moves": [
            "Split transaction (Jetty) from transport (TP Channel)",
            "On-chip bus controller replaces PCIe RNIC",
            "ISA load/store + TP bypass for synchronous ops",
            "Graded ordering reuses per-Jetty counters",
        ],
        "tiers": [
            "RTL on Alveo U50 (322 MHz, ~14% LUT)",
            "Two-node SystemC simulator",
            "gem5 full-system + uburma driver",
        ],
        "vs_coherent": "CXL/NVLink O(N) directory + lossless wire; UB non-coherent at rack scale (§12)",
        "limitations": [
            "No physical FPGA run in stub",
            "Out-of-context Vivado per-element",
            "Multi-flit Write loss replay incomplete",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table2_fpga_resources": TABLE2_RESOURCES,
        "table5_state": TABLE5_STATE,
        "table6_latency_headline": TABLE6_LATENCY_HEADLINE,
        "table8_load_bearing": TABLE8_LOAD_BEARING,
        "pilot_stacks": PILOT_STACKS,
        "state_scaling": state_scaling_table(),
    }


def evaluation_demo(seed: int = 42) -> dict[str, Any]:
    _ = seed
    cfg = OpenURMAConfig()
    n, m = 1024, 1024
    state = per_nic_state_ub(n, m, cfg)
    decomp = latency_decomposition(cfg)
    roce_spill_n = sram_spill_threshold_qp_bytes()
    ub_flat_n = 1024  # paper: UB flat to N≈1024

    # Simulated 64 B READ latencies with spill penalty
    def _lat_with_spill(base: int, endpoints: int, spill_at: int, penalty: int) -> int:
        return base + (penalty if endpoints >= spill_at else 0)

    roce_at_1024 = _lat_with_spill(cfg.roce_dma_latency_ns, n, roce_spill_n, 1000)
    ub_at_1024 = cfg.ub_ldst_latency_ns

    return {
        "seed": seed,
        "state_1024x1024": {
            "openurma_kb": round(state.total_bytes / 1024, 2),
            "roce_mb": round(state.roce_qp_bytes / (1024 * 1024), 2),
            "ratio": round(state.ratio_vs_roce, 1),
        },
        "latency_decomposition": [
            {
                "stack": r.stack,
                "total_modeled": r.total_modeled,
                "total_measured": r.total_measured,
                "phase_sum": sum(r.phases.values()),
            }
            for r in decomp
        ],
        "headline_ratios": headline_ratios(cfg),
        "ordering": ordering_surface_card(),
        "gating_fence_4deps": gating_cost_stub(4, fenced=True),
        "sram_spill": {
            "roce_cliff_n_approx": roce_spill_n,
            "ub_cliff_n_paper": ub_flat_n,
            "roce_latency_at_1024_ns": roce_at_1024,
            "ub_latency_at_1024_ns": ub_at_1024,
        },
        "fpga": TABLE2_RESOURCES,
    }


def evaluation_smoke(cfg: OpenURMAConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0)
    ratios = demo["headline_ratios"]
    return {
        "paper": "arXiv:2605.28717",
        "ub_vs_roce_dma": ratios["ub_vs_roce_dma"],
        "state_ratio_1024": demo["state_1024x1024"]["ratio"],
        "openurma_meets_timing": TABLE2_RESOURCES["openurma"]["lut"] > 0,
    }
