"""Leaf-peel topological generations and root-based baseline [3]."""

from __future__ import annotations

from typing import Any


def peel_generations_from_leaves(dag: dict[str, Any]) -> dict[int, int]:
    """Steps 4–6: Kahn peel in-degree-0 vertices; n=0 youngest (Sec. II.B)."""
    n_cracks: int = dag["n_cracks"]
    adj: dict[int, list[int]] = {k: list(v) for k, v in dag["adjacency"].items()}
    in_deg: dict[int, int] = dict(dag["in_degree"])
    remaining = set(range(n_cracks))
    generation: dict[int, int] = {}
    level = 0

    while remaining:
        leaves = [v for v in remaining if in_deg.get(v, 0) == 0]
        if not leaves:
            raise ValueError("DAG peel stalled — cycle or inconsistent hierarchy")
        for v in leaves:
            generation[v] = level
            remaining.remove(v)
        for v in leaves:
            for parent in adj.get(v, []):
                if parent in remaining:
                    in_deg[parent] = in_deg.get(parent, 0) - 1
        level += 1
    return generation


def peel_generations_from_roots(dag: dict[str, Any]) -> dict[int, int]:
    """Algorithm [3]: remove out-degree-0 (roots / oldest) first — unstable to boundary."""
    n_cracks: int = dag["n_cracks"]
    adj: dict[int, list[int]] = {k: list(v) for k, v in dag["adjacency"].items()}
    out_deg = {i: len(adj.get(i, [])) for i in range(n_cracks)}
    reverse: dict[int, list[int]] = {i: [] for i in range(n_cracks)}
    for child, parents in adj.items():
        for p in parents:
            reverse[p].append(child)

    remaining = set(range(n_cracks))
    generation: dict[int, int] = {}
    level = 0
    while remaining:
        roots = [v for v in remaining if out_deg.get(v, 0) == 0]
        if not roots:
            raise ValueError("root peel stalled")
        for v in roots:
            generation[v] = level
            remaining.remove(v)
        for v in roots:
            for child in reverse.get(v, []):
                if child in remaining:
                    out_deg[child] = out_deg.get(child, 0) - 1
        level += 1
    return generation


def peel_stages(dag: dict[str, Any], *, method: str = "leaf_peel") -> list[dict[str, Any]]:
    """Fig. 2/3 analogue: successive peel layers with crack ids per stage."""
    n_cracks: int = dag["n_cracks"]
    adj: dict[int, list[int]] = {k: list(v) for k, v in dag["adjacency"].items()}

    if method == "leaf_peel":
        in_deg: dict[int, int] = dict(dag["in_degree"])
        remaining = set(range(n_cracks))
        level = 0
        stages: list[dict[str, Any]] = []
        while remaining:
            layer = [v for v in remaining if in_deg.get(v, 0) == 0]
            if not layer:
                break
            stages.append({"level": level, "cracks": sorted(layer), "method": method})
            for v in layer:
                remaining.remove(v)
            for v in layer:
                for parent in adj.get(v, []):
                    if parent in remaining:
                        in_deg[parent] = in_deg.get(parent, 0) - 1
            level += 1
        return stages

    if method == "root_peel":
        out_deg = {i: len(adj.get(i, [])) for i in range(n_cracks)}
        reverse: dict[int, list[int]] = {i: [] for i in range(n_cracks)}
        for child, parents in adj.items():
            for p in parents:
                reverse[p].append(child)
        remaining = set(range(n_cracks))
        level = 0
        stages = []
        while remaining:
            layer = [v for v in remaining if out_deg.get(v, 0) == 0]
            if not layer:
                break
            stages.append({"level": level, "cracks": sorted(layer), "method": method})
            for v in layer:
                remaining.remove(v)
            for v in layer:
                for child in reverse.get(v, []):
                    if child in remaining:
                        out_deg[child] = out_deg.get(child, 0) - 1
            level += 1
        return stages

    raise ValueError(f"unknown method {method!r}")


def classify_crack_generations(dag: dict[str, Any], *, method: str = "leaf_peel") -> dict[str, Any]:
    if method == "leaf_peel":
        gen = peel_generations_from_leaves(dag)
    elif method == "root_peel":
        gen = peel_generations_from_roots(dag)
    else:
        raise ValueError(f"unknown method {method!r}")
    n_levels = max(gen.values()) + 1 if gen else 0
    return {
        "generations": gen,
        "n_generations": n_levels,
        "method": method,
        "youngest_cracks": [c for c, g in gen.items() if g == 0],
    }


def generation_stability_score(
    full: dict[int, int],
    fragment: dict[int, int],
    shared_cracks: set[int],
) -> float:
    """Fraction of shared cracks with identical generation under boundary shift."""
    if not shared_cracks:
        return 1.0
    matches = sum(1 for c in shared_cracks if full.get(c) == fragment.get(c))
    return matches / len(shared_cracks)
