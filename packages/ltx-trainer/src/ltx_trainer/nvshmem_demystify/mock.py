"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.nvshmem_demystify.paper import paper_card
from ltx_trainer.nvshmem_demystify.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    return {
        "paper": paper_card(),
        "demo": demo,
        "remote_offset_preserved": demo["remote_address_example"]["offset_preserved"],
        "on_stream_near_nccl_nvls": demo["on_stream_near_nccl_nvls"],
        "status": "ok" if demo["remote_address_example"]["offset_preserved"] else "fail",
    }
