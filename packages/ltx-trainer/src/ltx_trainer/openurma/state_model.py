"""Jetty / TP Channel vs Queue Pair state scaling (arXiv:2605.28717)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.openurma.config import OpenURMAConfig


@dataclass
class StateBreakdown:
    n_local: int
    m_remote: int
    jetty_table_bytes: int
    mr_table_bytes: int
    tp_channel_bytes: int
    total_bytes: int
    roce_qp_bytes: int
    ratio_vs_roce: float


def per_nic_state_ub(n: int, m: int, cfg: OpenURMAConfig | None = None) -> StateBreakdown:
    """O(N + M) — Jetty × N + MR × N + TP Channel × M."""
    cfg = cfg or OpenURMAConfig()
    jetty = n * cfg.jetty_bytes
    mr = n * cfg.mr_bytes
    tpc = m * cfg.tp_channel_bytes
    total = jetty + mr + tpc
    roce = n * m * cfg.roce_qp_bytes
    ratio = roce / total if total else float("inf")
    return StateBreakdown(
        n_local=n,
        m_remote=m,
        jetty_table_bytes=jetty,
        mr_table_bytes=mr,
        tp_channel_bytes=tpc,
        total_bytes=total,
        roce_qp_bytes=roce,
        ratio_vs_roce=ratio,
    )


def state_at_scale(n: int, m: int | None = None, cfg: OpenURMAConfig | None = None) -> dict[str, Any]:
    m = m if m is not None else n
    b = per_nic_state_ub(n, m, cfg)
    return {
        "n": n,
        "m": m,
        "openurma_bytes": b.total_bytes,
        "openurma_kb": round(b.total_bytes / 1024, 2),
        "roce_bytes": b.roce_qp_bytes,
        "roce_mb": round(b.roce_qp_bytes / (1024 * 1024), 2),
        "ratio": round(b.ratio_vs_roce, 1),
    }


def state_scaling_table(cfg: OpenURMAConfig | None = None) -> list[dict[str, Any]]:
    """Table 5 style symmetric N=M mesh."""
    cfg = cfg or OpenURMAConfig()
    rows = []
    for n in (1, 8, 64, 256, 1024):
        rows.append(state_at_scale(n, n, cfg))
    return rows


def sram_spill_threshold_qp_bytes(cache_kb: int = 256, qp_bytes: int = 512) -> int:
    """Approximate N where RoCE QPs exceed on-chip SRAM (~256 KB)."""
    cache_bytes = cache_kb * 1024
    # N^2 * qp_bytes > cache => N > sqrt(cache/qp)
    import math

    return max(1, int(math.ceil(math.sqrt(cache_bytes / qp_bytes))))
