"""CVSearch routing smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cvsearch.sgap import TreeNode


def toy_existence_confidences() -> dict[str, float]:
    return {"n0": 0.95, "n1": 0.6, "n2": 0.4}


def toy_tree_layer() -> list[TreeNode]:
    return [
        TreeNode("n0", depth=0, visual_complexity=0.9, bbox=(0.0, 0.0, 1.0, 1.0), pruned=False),
        TreeNode("n1", depth=0, visual_complexity=0.2, bbox=(0.0, 0.0, 0.5, 0.5), pruned=True),
        TreeNode("n2", depth=0, visual_complexity=0.5, bbox=(0.5, 0.0, 0.5, 0.5), pruned=False),
    ]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.cvsearch.search import expert_coverage_valid, information_sufficiency, route_search_mode

    conf = toy_existence_confidences()
    cq = information_sufficiency(conf["n0"])
    mode = route_search_mode(
        cq,
        expert_proposals_nonempty=True,
        expert_coverage_ok=expert_coverage_valid(2, 1),
        tau_q=0.5,
    )
    return {"num_tree_nodes": len(toy_tree_layer()), "cq_n0": round(cq, 4), "search_mode": mode.value}
