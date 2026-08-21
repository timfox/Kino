"""Collective algorithms, pSync, LL/LL128 (§VI)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.constants import TABLE2_COLLECTIVES


def collective_table() -> list[dict[str, Any]]:
    return [dict(row) for row in TABLE2_COLLECTIVES]


def psync_card() -> dict[str, Any]:
    return {
        "name": "pSync (persistent synchronization buffer)",
        "layout": "strided per team to avoid cache-line sharing",
        "subregions": "fixed offsets per collective; some double-buffered",
        "constraint": "one concurrent collective per team (shared internal state)",
    }


def ll_protocol_card() -> dict[str, Any]:
    return {
        "LL": "16-byte atomic write: 2 data + 2 flags; receiver polls flags",
        "LL128": "128-byte unit: 120 data + 8 flag; NVLink-only (128B atomics)",
        "purpose": "couple data movement with arrival notification for small messages",
    }


def multi_cta_card() -> dict[str, Any]:
    return {
        "device_api": "thread/warp/block scoped only; no public _grid variants",
        "host_on_stream": "FCollect, AllReduce, ReduceScatter multi-CTA via team_dups[]",
        "gate": "NVLS availability required for built-in multi-CTA path",
        "fallback": "NCCL when enabled; else single CTA",
    }


def barrier_card() -> dict[str, Any]:
    return {
        "algorithm": "dissemination over team pSync",
        "rounds": "log_k(N) with barrier_tg_dissem_kval",
        "sync_vs_barrier": "barrier adds quiet/threadfence before dissemination",
    }
