"""Reference HCP toy graph (Yang et al. Materials 18, 1067 Fig. 4a analogue)."""

from __future__ import annotations

from ltx_trainer.hcp_dag.graph import CrackGraph, Edge, build_crack_chains, build_crack_dag


def yang_fig4a_toy_graph() -> CrackGraph:
    """
    Minimal hierarchical T-junction network mirroring paper Fig. 1 topology.

    Layout (crossbar horizontal, legs downward):
        0---1---2---3
            |   |
            4   5
    """
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (3.0, 0.0),
        4: (1.0, -1.0),
        5: (2.0, -1.0),
    }
    edges = [
        Edge(0, 1),
        Edge(1, 2),
        Edge(2, 3),
        Edge(1, 4),
        Edge(2, 5),
    ]
    return CrackGraph(edges=edges, positions=positions)


def boundary_shifted_graph(drop_left: bool = False, drop_right: bool = False) -> CrackGraph:
    """Fragment variants for stability tests (Fig. 4 analogue)."""
    g = yang_fig4a_toy_graph()
    drop: set[int] = set()
    if drop_left:
        drop.update({0})
    if drop_right:
        drop.update({3})
    if not drop:
        return g
    keep_nodes = {n for n in g.positions if n not in drop}
    edges = [e for e in g.edges if e.u in keep_nodes and e.v in keep_nodes]
    positions = {n: g.positions[n] for n in keep_nodes}
    return CrackGraph(edges=edges, positions=positions)


def reference_pipeline() -> dict:
    graph = yang_fig4a_toy_graph()
    chains = build_crack_chains(graph)
    dag = build_crack_dag(chains, graph)
    return {"graph": graph, "chains": chains, "dag": dag}
