"""Benchmark kernel catalog (Table I) and grouping helpers."""

from __future__ import annotations

from typing import Any

from ltx_trainer.deopt_reopt.constants import KERNEL_GROUPS, TABLE1_KERNELS


def kernel_by_name(name: str) -> dict[str, Any]:
    for k in TABLE1_KERNELS:
        if k["name"] == name:
            return dict(k)
    raise KeyError(f"unknown kernel: {name}")


def kernels_by_group(group: str) -> list[dict[str, Any]]:
    if group not in KERNEL_GROUPS:
        raise ValueError(f"group must be one of {KERNEL_GROUPS}")
    return [dict(k) for k in TABLE1_KERNELS if k["group"] == group]


def kernel_catalog() -> list[dict[str, Any]]:
    return [dict(k) for k in TABLE1_KERNELS]


def divergent_kernels() -> list[str]:
    return [k["name"] for k in TABLE1_KERNELS if k["group"] == "divergent"]


def interpretive_group_note(kernel: str) -> str:
    """Author-defined interpretive framework from §III-D (not a predictor)."""
    k = kernel_by_name(kernel)
    group = str(k["group"])
    notes = {
        "divergent": "CPU optimizations obscure a different natural GPU mapping",
        "dominated": "Algorithmic dependences or numerical conventions dominate",
        "shared": "CPU structural hints remain useful on GPU",
    }
    return notes.get(group, "")
