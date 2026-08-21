"""Toy NEON-style forest scenes for executable demos."""

from __future__ import annotations

import random

from ltx_trainer.foresthg_trace.hypergraph import Hyperedge, SceneHypergraph, TreeNode
from ltx_trainer.foresthg_trace.operators import OperatorCall


def build_demo_scene(seed: int = 0) -> SceneHypergraph:
    """Paper-style slope poor-health ratio example (Figure 2)."""
    rng = random.Random(seed)
    nodes: dict[int, TreeNode] = {}
    nid = 0
    for slope in ("shady", "sunny"):
        n_trees = 138 if slope == "shady" else 111
        for _ in range(n_trees):
            health = rng.uniform(0.2, 0.95)
            if slope == "shady" and rng.random() < 0.26:
                health = rng.uniform(0.15, 0.49)
            if slope == "sunny" and rng.random() < 0.18:
                health = rng.uniform(0.15, 0.49)
            nodes[nid] = TreeNode(
                node_id=nid,
                health=health,
                height_m=rng.uniform(8.0, 28.0),
                slope_class=slope,
                x=rng.uniform(0, 500),
                y=rng.uniform(0, 500),
            )
            nid += 1
    # pad to paper totals ~532 / 612 all trees
    while len([n for n in nodes.values() if n.slope_class == "shady"]) < 532:
        nodes[nid] = TreeNode(
            node_id=nid,
            health=rng.uniform(0.3, 0.9),
            height_m=rng.uniform(5.0, 20.0),
            slope_class="shady",
            x=rng.uniform(0, 500),
            y=rng.uniform(0, 500),
        )
        nid += 1
    while len([n for n in nodes.values() if n.slope_class == "sunny"]) < 612:
        nodes[nid] = TreeNode(
            node_id=nid,
            health=rng.uniform(0.3, 0.9),
            height_m=rng.uniform(5.0, 20.0),
            slope_class="sunny",
            x=rng.uniform(0, 500),
            y=rng.uniform(0, 500),
        )
        nid += 1

    hyperedges = [
        Hyperedge("beh_all", "BEH", tuple(nodes.keys()), label="all_trees"),
        Hyperedge(
            "beh_shady",
            "BEH",
            tuple(n.node_id for n in nodes.values() if n.slope_class == "shady"),
            label="shady_unit",
        ),
        Hyperedge(
            "beh_sunny",
            "BEH",
            tuple(n.node_id for n in nodes.values() if n.slope_class == "sunny"),
            label="sunny_unit",
        ),
        Hyperedge(
            "reh_poor",
            "REH",
            tuple(n.node_id for n in nodes.values() if n.health < 0.5),
            label="poor_health_rule",
        ),
    ]
    return SceneHypergraph(scene_id="neon_demo_tile", nodes=nodes, hyperedges=hyperedges)


def gold_slope_ratio_program() -> list[OperatorCall]:
    return [
        OperatorCall("read", {"scope": "all_trees"}),
        OperatorCall("filter", {"field": "slope_class", "value": "shady", "group_key": "shady_all"}),
        OperatorCall("aggregate", {"metric": "poor_health_ratio", "store_as": "shady_ratio"}),
        OperatorCall("read", {"scope": "all_trees"}),
        OperatorCall("filter", {"field": "slope_class", "value": "sunny", "group_key": "sunny_all"}),
        OperatorCall("aggregate", {"metric": "poor_health_ratio", "store_as": "sunny_ratio"}),
        OperatorCall("compare", {"a": "shady_ratio", "b": "sunny_ratio"}),
        OperatorCall("audit", {}),
        OperatorCall("answer", {"template": "slope_ratio"}),
    ]
