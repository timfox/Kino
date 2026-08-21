"""OpenURMA / Unified Bus configuration (arXiv:2605.28717)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpenURMAConfig:
    """Reference parameters from paper evaluation defaults."""

    clock_mhz: float = 322.0
    link_delay_ns: int = 100
    payload_bytes: int = 64
    # Per-connection state (bytes)
    jetty_bytes: int = 20
    tp_channel_bytes: int = 56
    mr_bytes: int = 32
    roce_qp_bytes: int = 512
    # Headline 64 B remote fetch (Table 6 / §8.1)
    ub_ldst_latency_ns: int = 500
    ub_urma_latency_ns: int = 757
    roce_bf_latency_ns: int = 1686
    roce_dma_latency_ns: int = 2186
    # Pipeline cold paths (cycles @ 322 MHz)
    ub_ldst_tx_cycles: int = 8
    ub_urma_tx_cycles: int = 24
    roce_tx_cycles: int = 9
    # FPGA post-route (Table 2, Alveo U50)
    openurma_lut: int = 122_710
    openroce_lut: int = 46_636
    openurma_meets_322mhz: str = "38/38"
    # Throughput (§8.3)
    wr_rate_ub_mops: float = 2.80  # ratio vs OpenRoCE
    ycsb_throughput_ratio: float = 9.2
