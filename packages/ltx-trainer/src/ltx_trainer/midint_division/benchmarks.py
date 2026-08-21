"""Table 1 and summary anchors (A100 vs CGBN vs GMP)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.midint_division.constants import CGBN_DIV_MAX_BITS_EXP, TABLE1_ROWS
from ltx_trainer.midint_division.cost_model import div_to_mul_ratio


def table_1_cgbn_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in TABLE1_ROWS:
        bits_exp = int(r["bits_exp"])
        mul_ms = float(r["our_mul_ms"])
        div_mul = float(r["div_mul_ratio"])
        rows.append(
            {
                "bits_exp": bits_exp,
                "total_bits": 1 << bits_exp,
                "insts_exp": int(r["insts_exp"]),
                "parallel_instances": 1 << int(r["insts_exp"]),
                "our_mul_ms": mul_ms,
                "our_div_ms_est": round(mul_ms * div_mul, 3),
                "cgbn_mul_vs_ours": r["cgbn_mul_ratio"],
                "div_over_mul": div_mul,
                "cgbn_div_vs_ours": r["cgbn_div_ratio"],
                "gmp_speedup": r["gmp_speedup"],
                "cgbn_div_supported": bits_exp <= CGBN_DIV_MAX_BITS_EXP,
            }
        )
    return rows


def summary_anchors() -> dict[str, Any]:
    t1 = table_1_cgbn_comparison()
    best = t1[0]
    return {
        "hardware": "NVIDIA A100, RHEL 8.10, uint64, Q=4",
        "upstream": "github.com/aske0778/midint-arithmetic-division",
        "precision_range_bits_exp": (13, 18),
        "batch_constraint": "NumBits · NumInsts = 2^32",
        "cost_model_full_mults": "5–7 classical full multiplications per division",
        "best_div_mul_ratio": best["div_over_mul"],
        "best_div_mul_ratio_bits_exp": best["bits_exp"],
        "near_optimal_at_2p18": best["div_over_mul"] <= 5.5,
        "cgbn_div_max_bits_exp": CGBN_DIV_MAX_BITS_EXP,
        "cgbn_gap_at_2p15": 3.63,
    }


def positioning_vs_cgbn() -> dict[str, Any]:
    return {
        "cgbn": "warp-level cooperative groups; division to 2^15 bits in experiments",
        "midint_block": "one division per CUDA block; 2^13–2^18 bits",
        "complement": "block-level covers precisions above cgbn division support",
    }
