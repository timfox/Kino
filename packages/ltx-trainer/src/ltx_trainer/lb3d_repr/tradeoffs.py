"""Toy fidelity vs deployability scoring (Fig. 2 landscape)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lb3d_repr.taxonomy import landscape_nodes

_LEVEL = {"low": 0.0, "medium": 0.5, "high": 1.0}


def _score(level: str) -> float:
    return _LEVEL.get(level, 0.5)


def representation_tradeoff_report(*, seed: int = 0) -> dict[str, Any]:
    """Rank representations by a simple deployability index (no ML)."""
    del seed  # deterministic from literature anchors
    nodes = landscape_nodes()
    scored: list[dict[str, Any]] = []
    for node in nodes:
        fidelity = _score(str(node["fidelity"]))
        deploy = _score(str(node["deployability"]))
        memory_penalty = _score(str(node["memory"]))
        # Higher deploy + fidelity, lower memory → better index
        index = 0.45 * deploy + 0.35 * fidelity + 0.20 * (1.0 - memory_penalty)
        scored.append({**node, "tradeoff_index": round(index, 4)})
    scored.sort(key=lambda r: r["tradeoff_index"], reverse=True)
    best = scored[0]
    worst = scored[-1]
    return {
        "ranked": scored,
        "best_for_deployability": best["name"],
        "best_index": best["tradeoff_index"],
        "lowest_index": worst["tradeoff_index"],
        "hybrid_count": sum(1 for n in nodes if n["class"] == "hybrid"),
        "continuous_count": sum(1 for n in nodes if n["domain"] == "continuous"),
    }
