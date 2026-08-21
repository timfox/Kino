"""Visit-wise trajectory representation (Eq. 1–2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence


@dataclass(frozen=True)
class Visit:
    activity: str
    poi_id: str
    lat: float
    lon: float
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass(frozen=True)
class DailyTrajectory:
    individual_id: str
    date: str
    weekday: str
    visits: tuple[Visit, ...]

    @property
    def n_visits(self) -> int:
        return len(self.visits)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def daily_distance_km(visits: Sequence[Visit]) -> float:
    if len(visits) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(visits)):
        total += haversine_km(
            visits[i - 1].lat,
            visits[i - 1].lon,
            visits[i].lat,
            visits[i].lon,
        )
    return total


def radius_of_gyration_km(visits: Sequence[Visit]) -> float:
    if not visits:
        return 0.0
    lat_c = sum(v.lat for v in visits) / len(visits)
    lon_c = sum(v.lon for v in visits) / len(visits)
    return math.sqrt(
        sum(haversine_km(v.lat, v.lon, lat_c, lon_c) ** 2 for v in visits) / len(visits)
    )


def synthetic_poi_pool(seed: int = 42) -> dict[str, tuple[float, float]]:
    import random

    rng = random.Random(seed)
    pool: dict[str, tuple[float, float]] = {}
    for i in range(200):
        pool[f"poi_{i:03d}"] = (34.05 + rng.uniform(-0.15, 0.15), -118.25 + rng.uniform(-0.15, 0.15))
    return pool
