"""Planner helpers: density, swarm partitioning, MST trees (Sec. 3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ltx_trainer.swarical.config import FlsCameraOrientation, fls_density_per_area


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    def distance_to(self, other: Point3) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)


@dataclass
class FlSTreeEdge:
    child: int
    parent: int
    distance_cm: float
    child_orientation: FlsCameraOrientation


@dataclass
class SwarmTreeEdge:
    child_swarm: int
    parent_swarm: int
    primary_fls: int
    anchor_fls: int
    distance_cm: float


@dataclass
class PlannerOutput:
    """Stub planner artifact: point count, swarm count, tree edges."""

    n_fls: int
    n_swarms: int
    group_size_g: int
    fls_tree_edges: list[FlSTreeEdge] = field(default_factory=list)
    swarm_tree_edges: list[SwarmTreeEdge] = field(default_factory=list)
    dark_fls_inserted: int = 0


def estimate_fls_for_face_area(
    area: float,
    *,
    t_min: float,
    t_max: float,
    r: float,
) -> tuple[int, int]:
    """Minimum/maximum FLS count for a mesh face area (Sec. 3.1)."""
    d_min, d_max = fls_density_per_area(t_min=t_min, t_max=t_max, r=r)
    return max(1, int(math.floor(area * d_min))), max(1, int(math.ceil(area * d_max)))


def _mst_edges(indices: list[int], dist_fn) -> list[tuple[int, int, float]]:
    """Prim MST on small index sets (Sec. 3.2 swarm-tree / FLS-tree)."""
    if len(indices) <= 1:
        return []
    in_tree = {indices[0]}
    remaining = set(indices[1:])
    edges: list[tuple[int, int, float]] = []
    while remaining:
        best: tuple[int, int, float] | None = None
        for u in in_tree:
            for v in remaining:
                d = dist_fn(u, v)
                if best is None or d < best[2]:
                    best = (u, v, d)
        assert best is not None
        u, v, d = best
        edges.append((u, v, d))
        in_tree.add(v)
        remaining.remove(v)
    return edges


def build_fls_tree(
    points: list[Point3],
    *,
    primary_index: int = 0,
    max_link_cm: float,
) -> list[FlSTreeEdge]:
    """MST over FLS coordinates; BFS parent-child with line-of-sight orientation stub."""
    n = len(points)
    if n == 0:
        return []
    idx = list(range(n))
    edges = _mst_edges(idx, lambda i, j: points[i].distance_to(points[j]))
    parent_of = {primary_index: primary_index}
    for u, v, d in edges:
        if u not in parent_of:
            parent_of[u] = v
        elif v not in parent_of:
            parent_of[v] = u
    orientations = [FlsCameraOrientation.SIDE] * n
    out: list[FlSTreeEdge] = []
    for child, parent in parent_of.items():
        if child == parent:
            continue
        dz = points[child].z - points[parent].z
        if dz > 0.01:
            ori = FlsCameraOrientation.BOTTOM
        elif dz < -0.01:
            ori = FlsCameraOrientation.TOP
        else:
            ori = FlsCameraOrientation.SIDE
        orientations[child] = ori
        dist = points[child].distance_to(points[parent])
        if dist > max_link_cm:
            pass  # planner would insert dark FLS (Sec. 3.2)
        out.append(FlSTreeEdge(child, parent, dist, ori))
    return out


def swarm_centers(points: list[Point3], *, n_swarms: int) -> list[Point3]:
    """k-Means placeholder: split points into n_swarms contiguous chunks by x-coordinate."""
    if not points or n_swarms <= 0:
        return []
    sorted_pts = sorted(points, key=lambda p: (p.x, p.y, p.z))
    chunk = max(1, len(sorted_pts) // n_swarms)
    centers: list[Point3] = []
    for s in range(n_swarms):
        start = s * chunk
        end = len(sorted_pts) if s == n_swarms - 1 else min(len(sorted_pts), (s + 1) * chunk)
        group = sorted_pts[start:end]
        if not group:
            continue
        cx = sum(p.x for p in group) / len(group)
        cy = sum(p.y for p in group) / len(group)
        cz = sum(p.z for p in group) / len(group)
        centers.append(Point3(cx, cy, cz))
    return centers


def build_swarm_tree(centers: list[Point3]) -> list[SwarmTreeEdge]:
    """MST on swarm centers; primary/anchor = closest pair per edge (Sec. 3.2)."""
    n = len(centers)
    if n <= 1:
        return []
    idx = list(range(n))
    mst = _mst_edges(idx, lambda i, j: centers[i].distance_to(centers[j]))
    root = max(range(n), key=lambda i: sum(1 for u, v, _ in mst if u == i or v == i))
    parent_of = {root: root}
    for u, v, d in mst:
        if u not in parent_of:
            parent_of[u] = v
        elif v not in parent_of:
            parent_of[v] = u
    out: list[SwarmTreeEdge] = []
    for child, parent in parent_of.items():
        if child == parent:
            continue
        out.append(
            SwarmTreeEdge(
                child_swarm=child,
                parent_swarm=parent,
                primary_fls=child * 1000,
                anchor_fls=parent * 1000 + 1,
                distance_cm=centers[child].distance_to(centers[parent]),
            )
        )
    return out


def plan_from_points(
    points: list[Point3],
    *,
    group_size_g: int,
    max_link_cm: float = 8.0,
) -> PlannerOutput:
    """Minimal offline planner stub combining Steps 1–2."""
    from ltx_trainer.swarical.config import swarm_count

    n = len(points)
    n_g = max(1, swarm_count(n, group_size_g))
    centers = swarm_centers(points, n_swarms=n_g)
    fls_edges: list[FlSTreeEdge] = []
    chunk = max(1, n // n_g) if n_g else n
    dark = 0
    for s in range(n_g):
        start = s * chunk
        end = n if s == n_g - 1 else min(n, (s + 1) * chunk)
        group_pts = points[start:end]
        if len(group_pts) < 2:
            continue
        local = build_fls_tree(group_pts, primary_index=0, max_link_cm=max_link_cm)
        for e in local:
            fls_edges.append(
                FlSTreeEdge(
                    child=start + e.child,
                    parent=start + e.parent,
                    distance_cm=e.distance_cm,
                    child_orientation=e.child_orientation,
                )
            )
            if e.distance_cm > max_link_cm:
                dark += 1
    return PlannerOutput(
        n_fls=n,
        n_swarms=len(centers),
        group_size_g=group_size_g,
        fls_tree_edges=fls_edges,
        swarm_tree_edges=build_swarm_tree(centers),
        dark_fls_inserted=dark,
    )
