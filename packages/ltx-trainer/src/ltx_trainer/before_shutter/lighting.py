"""Controllable portrait lighting (Sec. 3.1, 3.5)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState


@dataclass
class LightDevice:
    name: str
    power_w: float
    distance_m: float
    color_temp_k: float = 5600.0
    emissive: bool = True


PRESET_LAYOUTS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "rembrandt": (("key", 620.0, 1.0), ("fill", 100.0, 1.5), ("rim", 80.0, 2.0)),
    "split": (("key", 500.0, 1.2), ("fill", 25.0, 2.0)),
    "butterfly": (("key", 400.0, 0.8), ("fill", 120.0, 1.6)),
    "loop": (("key", 350.0, 1.1), ("fill", 150.0, 1.7)),
    "rim_key_fill": (("key", 300.0, 1.0), ("fill", 100.0, 1.4), ("rim", 200.0, 2.2)),
    "chiaroscuro": (("key", 800.0, 0.9), ("fill", 15.0, 2.5), ("negative_fill", 0.0, 1.0)),
}


def lighting_ratio(key_power: float, fill_power: float) -> float:
    """Portrait lighting ratio key:fill."""
    return key_power / max(fill_power, 1e-3)


def apply_preset(state: PortraitPlanState, preset: str) -> PortraitPlanState:
    layout = PRESET_LAYOUTS.get(preset, PRESET_LAYOUTS["rembrandt"])
    powers = tuple(x[1] for x in layout)
    dists = tuple(x[2] for x in layout)
    state.preset = preset
    state.light_powers_w = powers
    state.light_distances_m = dists
    return state


def refine_lighting_step(
    state: PortraitPlanState,
    *,
    critique: str,
    cfg: BeforeShutterConfig | None = None,
) -> PortraitPlanState:
    """One Photographer lighting refinement from Judge critique (stub)."""
    _ = cfg
    c = critique.lower()
    powers = list(state.light_powers_w)
    dists = list(state.light_distances_m)
    if "negative fill" in c or "contrast" in c:
        if len(powers) >= 2:
            powers[1] = max(10.0, powers[1] * 0.5)
    if "harsh" in c or "wash" in c:
        powers[0] = max(50.0, powers[0] * 0.85)
        state.exposure_comp_stops -= 0.3
    if "underexposed" in c or "dark" in c:
        state.exposure_comp_stops += 0.5
    if "rim" in c and len(powers) >= 3:
        powers[2] = min(400.0, powers[2] * 1.2)
    if "distance" in c and dists:
        dists[0] = max(0.5, dists[0] - 0.1)
    state.light_powers_w = tuple(powers)
    state.light_distances_m = tuple(dists)
    return state


def devices_from_state(state: PortraitPlanState) -> list[LightDevice]:
    layout = PRESET_LAYOUTS.get(state.preset, PRESET_LAYOUTS["rembrandt"])
    out: list[LightDevice] = []
    for (name, _p, _d), p, d in zip(layout, state.light_powers_w, state.light_distances_m):
        out.append(LightDevice(name=name, power_w=p, distance_m=d, emissive=name != "negative_fill"))
    return out
