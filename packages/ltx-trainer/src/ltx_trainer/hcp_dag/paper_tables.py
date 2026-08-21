"""Paper anchors and knowledge card (Tarasevich et al., arXiv:2606.03473)."""

from __future__ import annotations

from typing import Any


def knowledge_card() -> dict[str, Any]:
    return {
        "paper": "arXiv:2606.03473",
        "title": "Hierarchical crack patterns: Identification of crack generations",
        "authors": ["Yuri Yu. Tarasevich", "Andrei S. Burmistrov", "Andrei V. Eserkepov"],
        "task": "Classify spatial crack generations in an HCP image fragment via DAG topological sort",
        "physics_anchors": [
            "Cracks = edge chains with consecutive angles α ∈ [150°, 180°] (Sec. II.A)",
            "T-junction: leg (child) → crossbar (parent); DAG arcs child→parent",
            "Youngest generation = DAG leaves (in-degree 0); peel via Kahn [9]",
            "Contrasts root-first algorithm [3] (Bohn et al., Phys. Rev. E 71, 046214)",
            "Avoids angle-based fragmentation of Kumar & Kulkarni [8]",
            "Reference HCP: Yang et al. Materials 18, 1067 (2025) Fig. 4a",
        ],
        "limitations": [
            "Single component, vertex degree ≤ 3, no multi-edge loops",
            "Strictly hierarchical — X-junctions and Y-stars excluded",
            "Temporal hierarchy not recoverable from final image alone",
        ],
    }


def algorithm_steps() -> list[str]:
    return [
        "Find crack chains (α ∈ [150°, 180°] between consecutive edges)",
        "Identify leaves (youngest): T at both ends, or T + boundary degree 1",
        "Build DAG: arcs from leg crack to crossbar crack at each T-junction",
        "Assign n=0 to in-degree-0 vertices; peel leaves and increment n",
        "Repeat until DAG empty (topological sort by leaf removal)",
    ]


def comparison_algorithms() -> dict[str, str]:
    return {
        "leaf_peel": "Proposed: remove in-degree-0 (youngest) — stable to boundary shifts",
        "root_peel": "Bohn et al. [3]: remove out-degree-0 (oldest roots) — unstable",
        "angle_fragmented": "Kumar & Kulkarni [8]: junction angles — fragments single cracks",
    }


def reference_anchors() -> dict[str, Any]:
    return {
        "chain_angle_min_deg": 150.0,
        "chain_angle_max_deg": 180.0,
        "max_vertex_degree": 3,
        "yang_fig4a_toy_n_cracks": 3,
        "yang_fig4a_toy_n_generations": 2,
        "leaf_peel_boundary_stability_min": 0.9,
    }
