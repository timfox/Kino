"""Paper benchmark anchors (Tables 1–2, Figs 11–19)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pccl.constants import (
    ALL_TO_ALL_COMPLEXITY,
    PROCESS_GROUP_SPEEDUP_AVG,
    SYNTHESIS_512_NPU_MINUTES,
    SYNTHESIS_1000_NPU_HOURS,
    SYNTHESIZER_COMPARISON,
    TE_CCL_SPEEDUP_36NPU,
)


def table_i_synthesizer_comparison() -> list[dict[str, Any]]:
    rows = []
    for name, scalable, gen_topo, gen_coll, pg_aware in SYNTHESIZER_COMPARISON:
        rows.append(
            {
                "synthesizer": name,
                "scalable": scalable,
                "generic_topology": gen_topo,
                "generic_collective": gen_coll,
                "process_group_aware": pg_aware,
            }
        )
    return rows


def table_ii_collective_support() -> dict[str, dict[str, bool]]:
    patterns = ("reduce_scatter", "all_gather", "all_reduce", "all_to_all")
    support = {
        "SCCL": {p: False for p in patterns},
        "TACCL": {p: False for p in patterns},
        "Blink": {"reduce_scatter": False, "all_gather": False, "all_reduce": True, "all_to_all": False},
        "MultiTree": {"reduce_scatter": True, "all_gather": True, "all_reduce": True, "all_to_all": False},
        "ForestColl": {"reduce_scatter": True, "all_gather": True, "all_reduce": True, "all_to_all": False},
        "TACOS": {"reduce_scatter": True, "all_gather": True, "all_reduce": True, "all_to_all": False},
        "TE-CCL": {p: (p == "all_to_all") for p in patterns},
        "PCCL": {p: True for p in patterns},
    }
    return support


def scalability_anchors() -> dict[str, Any]:
    return {
        "all_to_all_complexity": ALL_TO_ALL_COMPLEXITY,
        "synthesis_512_npu_minutes": SYNTHESIS_512_NPU_MINUTES,
        "synthesis_1000_npu_hours": SYNTHESIS_1000_NPU_HOURS,
        "pccl_vs_te_ccl_36npu_speedup": TE_CCL_SPEEDUP_36NPU,
        "fig_11_note": "PCCL >3 orders of magnitude faster than TE-CCL at 36-NPU 2D mesh",
    }


def process_group_speedup_anchors() -> dict[str, Any]:
    return {
        "avg_speedup_vs_direct": PROCESS_GROUP_SPEEDUP_AVG,
        "fig_16_range": "2.33–3.03×",
        "fig_19_single_group": 3.05,
        "fig_17_two_groups": 2.8,
        "fig_18_partial_group": 1.88,
    }


def parallelism_collectives_table() -> dict[str, dict[str, bool]]:
    """Table 3: collectives required by parallelization strategy."""
    return {
        "data": {"reduce_scatter": False, "all_gather": False, "all_reduce": True, "all_to_all": False},
        "tensor": {"reduce_scatter": True, "all_gather": True, "all_reduce": False, "all_to_all": False},
        "expert": {"reduce_scatter": False, "all_gather": False, "all_reduce": False, "all_to_all": True},
        "pipeline": {"reduce_scatter": False, "all_gather": False, "all_reduce": False, "all_to_all": False, "pt_to_pt": True},
    }
