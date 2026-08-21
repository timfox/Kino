"""Bidirectional latency cost model for 64 B READ (Table 7, arXiv:2605.28717)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.openurma.config import OpenURMAConfig


@dataclass
class LatencyRow:
    stack: str
    phases: dict[str, int]
    total_modeled: int
    total_measured: int | None = None


def _ub_ldst_phases(cfg: OpenURMAConfig) -> dict[str, int]:
    return {
        "submit_membus": 30,
        "nic_tx": 25,
        "wire_forward": cfg.link_delay_ns,
        "target_nic_rx": 25,
        "target_nic_dram": 30,
        "target_dram_hit": 30,
        "target_nic_tx": 25,
        "wire_back": cfg.link_delay_ns,
        "initiator_nic_rx": 25,
        "complete_membus": 30,
    }


def _ub_urma_phases(cfg: OpenURMAConfig) -> dict[str, int]:
    return {
        "verb_post": 50,
        "wqe_construct": 30,
        "submit_membus": 30,
        "nic_tx": 78,
        "wire_forward": cfg.link_delay_ns,
        "target_nic_rx": 78,
        "target_nic_dram": 30,
        "target_dram_hit": 30,
        "target_nic_tx": 78,
        "wire_back": cfg.link_delay_ns,
        "initiator_nic_rx": 78,
        "complete_membus": 30,
        "cqe_poll": 5,
        "verb_poll": 30,
    }


def _roce_dma_phases(cfg: OpenURMAConfig) -> dict[str, int]:
    return {
        "verb_post": 50,
        "wqe_construct": 30,
        "doorbell_mmio": 150,
        "dma_wqe_fetch": 500,
        "nic_tx": 28,
        "wire_forward": cfg.link_delay_ns,
        "target_nic_rx": 28,
        "target_dma_read": 500,
        "target_dram_hit": 30,
        "target_nic_tx": 28,
        "wire_back": cfg.link_delay_ns,
        "initiator_nic_rx": 28,
        "initiator_resp_dma": 250,
        "dma_cqe_write": 250,
        "cqe_poll": 70,
        "verb_poll": 30,
    }


def latency_decomposition(cfg: OpenURMAConfig | None = None) -> list[LatencyRow]:
    cfg = cfg or OpenURMAConfig()
    rows = [
        LatencyRow("UB LD/ST", _ub_ldst_phases(cfg), 420, cfg.ub_ldst_latency_ns),
        LatencyRow("UB URMA", _ub_urma_phases(cfg), 745, cfg.ub_urma_latency_ns),
        LatencyRow("RoCE DMA", _roce_dma_phases(cfg), 2172, cfg.roce_dma_latency_ns),
    ]
    return rows


def headline_ratios(cfg: OpenURMAConfig | None = None) -> dict[str, float]:
    cfg = cfg or OpenURMAConfig()
    ub = cfg.ub_ldst_latency_ns
    return {
        "ub_vs_roce_dma": round(cfg.roce_dma_latency_ns / ub, 2),
        "ub_vs_roce_bf": round(cfg.roce_bf_latency_ns / ub, 2),
        "ub_urma_vs_roce_dma": round(cfg.roce_dma_latency_ns / cfg.ub_urma_latency_ns, 2),
    }


def stacks_summary(cfg: OpenURMAConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or OpenURMAConfig()
    return [
        {"stack": "UB LD/ST", "latency_ns": cfg.ub_ldst_latency_ns, "path": "ISA load + TP bypass"},
        {"stack": "UB URMA", "latency_ns": cfg.ub_urma_latency_ns, "path": "membus doorbell + WR"},
        {"stack": "RoCE BF", "latency_ns": cfg.roce_bf_latency_ns, "path": "inline WQE ≤64 B"},
        {"stack": "RoCE DMA", "latency_ns": cfg.roce_dma_latency_ns, "path": "PCIe WQE fetch"},
    ]
