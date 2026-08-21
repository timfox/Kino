"""Scope notes for Swarical reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Raspberry Pi / ArUco capture requires upstream aruco-pose-estimation on hardware; GOPEX provides range models and commands.",
    "Small-scale 16-FLS localization is a deterministic correction stub, not UDP multicast processes.",
    "k-Means swarm partitioning uses x-sort chunking placeholder, not Lloyd's algorithm.",
    "Mesh→Poisson-disk sampling and thousand-core log replay use upstream Swarical Docker artifacts.",
    "SwarMer online merge baseline is not simulated; comparison constants come from paper Sec. 5.4.",
)
