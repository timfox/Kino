"""CPU smoke exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fp8_ozaki.paper import paper_card
from ltx_trainer.fp8_ozaki.pipeline import run_demo


def evaluation_smoke() -> dict[str, Any]:
    demo = run_demo()
    gemm = demo["workload_speedups"].get("dense_gemm", 0.0)
    return {
        "paper": paper_card(),
        "demo": demo,
        "b300_gemm_speedup_ok": gemm >= 100.0,
        "status": "ok",
    }
