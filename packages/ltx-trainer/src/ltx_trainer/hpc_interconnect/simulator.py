"""Analytic performance-ratio model aligned to paper anchors (Fig. 4–6)."""

from __future__ import annotations

import math

from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, FabricProfile, SystemName, get_system


def predict_performance_ratio(
    system: SystemName | str,
    *,
    nodes: int,
    aggressor: AggressorPattern | str,
    message_bytes: int = 32 * 1024,
    steady: bool = True,
    burst_pause_collectives: int | None = None,
    burst_length_collectives: int | None = None,
    nslb_enabled: bool | None = None,
) -> float:
    """
    Ratio uncongested_runtime / congested_runtime (higher is better; paper heatmaps).

    Tuned so CRESCO8/Leonardo/LUMI ordering matches Fig. 5–6 at 64–256 nodes.
    """
    profile = get_system(system)
    agg = AggressorPattern(aggressor)
    n = max(4, nodes)
    if nslb_enabled is None:
        nslb_enabled = profile.nslb_enabled

    scale = (n / 64.0) ** profile.scale_penalty
    frac = profile.partition_fraction(n)
    part_penalty = 1.0 - profile.partition_fraction_penalty * frac

    msg_factor = 1.0 - 0.05 * math.log2(max(8, message_bytes) / 512.0) / 16.0

    if agg == AggressorPattern.ALLTOALL:
        resilience = profile.alltoall_resilience
        transit_stress = scale * (1.1 if profile.name == SystemName.CRESCO8 else 0.7)
    else:
        resilience = profile.incast_resilience
        transit_stress = scale * (1.4 if profile.name == SystemName.LEONARDO else 0.9)

    ratio = resilience * part_penalty * msg_factor
    ratio -= (1.0 - resilience) * min(0.85, transit_stress)

    if profile.name == SystemName.NANJING and not nslb_enabled and agg == AggressorPattern.ALLTOALL:
        ratio *= 120.0 / 180.0

    if profile.name == SystemName.HAICGU:
        ratio *= 0.85

    if not steady and burst_pause_collectives is not None:
        pause = max(0, burst_pause_collectives)
        blen = burst_length_collectives or 4
        duty = blen / max(1, blen + pause)
        gap = min(1.0, pause / 4.0)
        recovery = profile.bursty_recovery * gap
        ratio = ratio + (1.0 - ratio) * (1.0 - recovery) * duty
        if agg == AggressorPattern.INCAST and pause <= 1:
            ratio *= 0.55

    return float(max(0.05, min(1.05, ratio)))


def compare_to_anchor(
    system: SystemName | str,
    scenario_key: str,
    predicted: float,
    *,
    rel_tol: float = 0.35,
) -> dict[str, float | bool | str]:
    from ltx_trainer.hpc_interconnect.benchmarks import FIG5_STEADY_ANCHORS, FIG6_BURSTY_64

    anchors = {**FIG5_STEADY_ANCHORS, **FIG6_BURSTY_64}
    ref = anchors.get(scenario_key)
    if ref is None:
        return {"scenario": scenario_key, "predicted": predicted, "matched": True}
    target = ref.get("typical", ref.get("min_ratio", predicted))
    matched = abs(predicted - target) <= rel_tol * max(0.1, target)
    return {
        "scenario": scenario_key,
        "predicted": predicted,
        "reference": target,
        "matched": matched,
    }
