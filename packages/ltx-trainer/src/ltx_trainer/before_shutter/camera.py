"""Thin-lens camera + exposure (Sec. 3.1, Eq. 2–3, 8–9)."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor

from ltx_trainer.before_shutter.config import BeforeShutterConfig, PortraitPlanState


def ev100_from_aperture_shutter(f_number: float, shutter_sec: float) -> float:
    """EV100 = log2(N_f^2 / tau) — paper Eq. (2)."""
    return math.log2((f_number**2) / max(shutter_sec, 1e-9))


def meter_shutter_from_ev100(f_number: float, ev100: float) -> float:
    return (f_number**2) / (2.0**ev100)


def render_linear_radiance(
    base_luminance: Tensor,
    *,
    ev100: float,
    exposure_comp_stops: float,
    kappa: float = 1.0,
) -> Tensor:
    """Simplified Eq. (3): linear exposure then monotone saturate."""
    gain = kappa * (2.0 ** (-(ev100 + exposure_comp_stops)))
    exposed = base_luminance * gain
    return exposed / (1.0 + exposed.clamp(min=0.0))


def exposure_validity_logit(
    image_linear: Tensor,
    *,
    cfg: BeforeShutterConfig | None = None,
) -> tuple[float, float]:
    """p_valid fraction and V_exp logit — Eq. (8–9)."""
    cfg = cfg or BeforeShutterConfig()
    s = cfg.exposure_stops
    mid = image_linear.median().clamp(min=1e-6)
    rho = image_linear / mid
    valid = ((rho >= 2.0 ** (-s)) & (rho <= 2.0**s)).float().mean().item()
    eps = 1e-6
    v_exp = math.log((valid + eps) / (1.0 - valid + eps))
    return valid, v_exp


def plan_camera_from_graph_constraint(
    state: PortraitPlanState,
    *,
    keep_nodes: Iterable[str],
    ev_targets: dict[str, float],
) -> PortraitPlanState:
    """Heuristic composition revision ΔC from SG visibility + EV latitude."""
    _ = keep_nodes
    avg_ev = sum(ev_targets.values()) / max(len(ev_targets), 1)
    delta = max(-1.0, min(1.0, -avg_ev * 0.15))
    state.exposure_comp_stops = round(state.exposure_comp_stops + delta, 2)
    state.camera_extrinsic = (
        state.camera_extrinsic[0],
        state.camera_extrinsic[1] + 0.05 * delta,
        state.camera_extrinsic[2],
    )
    return state
