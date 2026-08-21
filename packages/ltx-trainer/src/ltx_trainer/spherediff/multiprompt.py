"""Multi-prompt elevation mapping (Sec. 3.5)."""

from __future__ import annotations

import math

from ltx_trainer.spherediff.config import ELEVATION_PROMPTS


def elevation_deg(d: tuple[float, float, float]) -> float:
    x, y, z = d
    return math.degrees(math.asin(max(-1.0, min(1.0, y))))


def select_prompt_index(elev: float) -> int:
    """λ_i: nearest discrete elevation in {-90,-10,0,10,90}."""
    keys = ELEVATION_PROMPTS
    return min(range(len(keys)), key=lambda i: abs(elev - keys[i]))


def prompt_for_direction(direction: tuple[float, float, float], prompts: list[str]) -> str:
    idx = select_prompt_index(elevation_deg(direction))
    return prompts[min(idx, len(prompts) - 1)]


def view_direction_unit(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit direction on S² from paper azimuth θ and elevation φ."""
    th = math.radians(azimuth_deg)
    ph = math.radians(elevation_deg)
    x = math.sin(th) * math.cos(ph)
    y = math.sin(ph)
    z = math.cos(th) * math.cos(ph)
    return (x, y, z)
