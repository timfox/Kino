"""Semantic scattering structure S=(K,A,G) — Sec. II."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ltx_trainer.s3u_sar.constants import (
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    VISIBILITY_DEGRADED,
    VISIBILITY_INVALID,
    VISIBILITY_SALIENT,
)


@dataclass
class SemanticKeypoint:
    x: float
    y: float
    component: str
    visibility: int


def visibility_label(vsem: bool, vscat: bool) -> int:
    """Eq. 3 visibility-aware scattering semantics."""
    if not vsem:
        return VISIBILITY_INVALID
    return VISIBILITY_SALIENT if vscat else VISIBILITY_DEGRADED


# Physics-constrained rigid-body topology edges (Fig. 5 — simplified skeleton)
DEFAULT_TOPOLOGY_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),  # nose
    (1, 4),
    (1, 5),  # fuselage to wing roots
    (4, 2),
    (5, 3),  # wing root to tip
    (1, 6),
    (6, 7),  # tail
    (4, 8),
    (5, 9),  # engines
)


def valid_topology_edges(
    visibilities: list[int],
    edges: tuple[tuple[int, int], ...] | None = None,
) -> list[tuple[int, int]]:
    """Visibility-guided valid edge set M — Eq. 13."""
    edges = edges or DEFAULT_TOPOLOGY_EDGES
    return [
        (i, j)
        for i, j in edges
        if visibilities[i] >= VISIBILITY_DEGRADED and visibilities[j] >= VISIBILITY_DEGRADED
    ]


def structure_representation_card() -> dict[str, Any]:
    return {
        "formulation": "S = (K, A, G)",
        "keypoints": f"N={NUM_KEYPOINTS} semantic scattering keypoints with component labels",
        "visibility": "vi=2 salient, vi=1 degraded, vi=0 invalid (Eq. 3)",
        "topology": "G=(V,E) rigid-body aircraft graph (Eq. 4)",
        "components": list(KEYPOINT_NAMES),
    }
