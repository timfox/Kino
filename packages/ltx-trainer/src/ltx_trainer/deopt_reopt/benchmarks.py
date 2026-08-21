"""Table I–II reproduction and positioning summary."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.constants import MODEL_O120, MODEL_Q235, TABLE1_KERNELS
from ltx_trainer.deopt_reopt.kernels import kernel_catalog
from ltx_trainer.deopt_reopt.statistics import (
    iterative_significance_summary,
    single_shot_significance_summary,
    table2_rows,
)


def table_1_kernels() -> list[dict[str, Any]]:
    return kernel_catalog()


def table_2_comparison(workflow: str = "single_shot") -> list[dict[str, Any]]:
    return table2_rows(workflow)


def conv2d_highlight() -> dict[str, Any]:
    """conv2d: clearest divergent-style D+R win on both models (Single-shot)."""
    ss = table2_rows("single_shot")
    rows = {f"{r['model']}_{r['kernel']}": r for r in ss}
    o120 = rows["O120_conv2d"]
    q235 = rows["Q235_conv2d"]
    return {
        "kernel": "conv2d",
        "group": "divergent",
        "single_shot": {
            "O120": {
                "d_succ_pct": o120["d_succ"],
                "dr_succ_pct": o120["dr_succ"],
                "dr_over_d_median": o120.get("dr_over_d_ratio"),
                "bh_significant": True,
            },
            "Q235": {
                "d_succ_pct": q235["d_succ"],
                "dr_succ_pct": q235["dr_succ"],
                "dr_over_d_median": q235.get("dr_over_d_ratio"),
                "bh_significant": True,
            },
        },
        "interpretation": "Filter-transpose cache + NEON blocking obscures GPU tiling strategy",
    }


def summary_anchors() -> dict[str, Any]:
    return {
        "hardware": "GH200 + H100, NVHPC 25.9, CUDA 12.6",
        "models": (MODEL_O120, MODEL_Q235),
        "kernels": len(TABLE1_KERNELS),
        "upstream": "github.com/mukunoki/deopt_reopt",
        "single_shot": single_shot_significance_summary(),
        "iterative": iterative_significance_summary(),
        "conv2d": conv2d_highlight(),
        "headline": "Deopt-Reopt effective but non-universal; kernel/model/budget dependent",
    }


def positioning_vs_direct() -> dict[str, Any]:
    return {
        "direct": "2 LLM calls: translate optimized C++ → CUDA → reoptimize",
        "deopt_reopt": "3 LLM calls: deoptimize → translate → reoptimize",
        "when_helps": "divergent-style kernels (conv2d), Q235 feasibility on CPU-anchored models",
        "when_hurts": "dependency-dominated kernels where CPU structure helps (O120 ddgemm, btdma)",
        "iterative_effect": "narrows O120 gap; Q235 retains large wins on conv2d/ddgemm/bgemm",
    }
