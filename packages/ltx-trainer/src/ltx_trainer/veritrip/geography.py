"""Geographic coherence: TSP reference distance and AM metric (Sec. 3.4)."""

from __future__ import annotations

import math
from itertools import permutations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def route_length_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += haversine_km(*coords[i], *coords[i + 1])
    return total


def tsp_optimal_km(coords: list[tuple[float, float]]) -> float:
    """Exact TSP for small POI sets; falls back to nearest-neighbor for n > 8."""
    n = len(coords)
    if n <= 1:
        return 0.0
    if n <= 8:
        best = float("inf")
        for perm in permutations(range(n)):
            path = [coords[i] for i in perm]
            d = route_length_km(path)
            best = min(best, d)
        return best
    return tsp_nearest_neighbor_km(coords)


def tsp_nearest_neighbor_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) <= 1:
        return 0.0
    unvisited = list(range(len(coords)))
    start = unvisited.pop(0)
    path = [coords[start]]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: haversine_km(*coords[current], *coords[j]))
        unvisited.remove(nxt)
        path.append(coords[nxt])
        current = nxt
    return route_length_km(path)


def average_margin(
    agent_km: float,
    reference_km: float,
    num_pois: int,
    *,
    scale_10km: float = 10.0,
) -> float:
    """
    AM ↓ = max(0, D_agent - D_ref) / len(POIs), reported in units of 10 km (Table 4).
    """
    if num_pois <= 0:
        return 0.0
    excess = max(0.0, agent_km - reference_km)
    return (excess / num_pois) / scale_10km
