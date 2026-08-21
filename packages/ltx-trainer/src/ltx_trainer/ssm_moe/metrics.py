"""Paper table anchors for SSM + MoE variants."""

from __future__ import annotations

from typing import Any


def table_moe_mamba_scaling() -> dict[str, Any]:
    """MoE-Mamba (arXiv:2401.04081) training-step efficiency vs dense Mamba."""
    return {
        "paper": "arXiv:2401.04081",
        "metric": "steps_to_match_val_ppl",
        "mamba_dense": 1.0,
        "moe_mamba": 0.45,
        "note": "MoE-Mamba reaches Mamba val PPL in ~2.2x fewer steps (paper Fig. 2).",
    }


def table_swimba_benchmarks() -> dict[str, Any]:
    """Swimba (arXiv:2603.06938) matched-FLOP average vs Mamba-2 baseline."""
    return {
        "paper": "arXiv:2603.06938",
        "tasks": {
            "pile_ppl": {"mamba2": 1.0, "swimba": 0.98},
            "lamba_avg": {"mamba2": 1.0, "swimba": 1.02},
        },
        "single_trajectory": True,
        "latency_overhead_pct": 8.0,
    }


def table_routing_mamba_scaling() -> dict[str, Any]:
    """Routing Mamba (arXiv:2506.18145) active vs total parameters."""
    return {
        "paper": "arXiv:2506.18145",
        "active_params_B": 1.3,
        "total_params_B": 10.0,
        "dense_mamba_equiv_active_B": 3.0,
        "flops_saving_pct": 23.0,
    }


def table_mossnet_lm() -> dict[str, Any]:
    """MossNet (arXiv:2510.26182) SSM-expert heads vs Transformer baseline."""
    return {
        "paper": "arXiv:2510.26182",
        "params_M": 350,
        "tokens_B": 300,
        "mossnet_ppl": 12.4,
        "transformer_ppl": 13.1,
        "ssm_single_head_ppl": 13.8,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "moe_mamba": table_moe_mamba_scaling(),
        "swimba": table_swimba_benchmarks(),
        "routing_mamba": table_routing_mamba_scaling(),
        "mossnet": table_mossnet_lm(),
    }
