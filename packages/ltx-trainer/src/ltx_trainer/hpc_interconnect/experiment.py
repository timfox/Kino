"""Run victim benchmark under steady/bursty congestion (methodology stub)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.hpc_interconnect.collectives import (
    VictimCollective,
    aggregate_runtime_ms,
    collective_cost_us,
    simulate_iterations,
)
from ltx_trainer.hpc_interconnect.congestion import BurstyCongestion, SteadyCongestion
from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, FabricProfile, SystemName, get_system
from ltx_trainer.hpc_interconnect.simulator import compare_to_anchor, predict_performance_ratio


def run_congestion_experiment(
    system: SystemName | str,
    *,
    nodes: int = 64,
    victim: VictimCollective | str = VictimCollective.ALLGATHER_RING,
    aggressor: AggressorPattern | str = AggressorPattern.ALLTOALL,
    message_bytes: int = 32 * 1024,
    steady: bool = True,
    burst_pause_collectives: int = 2,
    burst_length_collectives: int = 4,
    link_gbps: float = 100.0,
    iterations: int = 200,
) -> dict[str, Any]:
    profile = get_system(system)
    agg = AggressorPattern(aggressor)
    vic = VictimCollective(victim)

    base_us = collective_cost_us(vic, nodes=nodes, message_bytes=message_bytes, per_link_gbps=link_gbps)

    if steady:
        inj = SteadyCongestion(aggressor=agg)
        slowdown = inj.slowdown_multiplier(profile, nodes=nodes)
    else:
        inj = BurstyCongestion(
            aggressor=agg,
            burst_collectives=burst_length_collectives,
            pause_collectives=burst_pause_collectives,
        )
        slowdown = inj.slowdown_multiplier(profile, nodes=nodes)

    uncongested = simulate_iterations(base_us, iterations=iterations, congestion_factor=1.0)
    congested = simulate_iterations(base_us, iterations=iterations, congestion_factor=slowdown)

    t_base = aggregate_runtime_ms(uncongested)
    t_cong = aggregate_runtime_ms(congested)
    measured_ratio = t_base / max(1e-9, t_cong)
    predicted_ratio = predict_performance_ratio(
        profile.name,
        nodes=nodes,
        aggressor=agg,
        message_bytes=message_bytes,
        steady=steady,
        burst_pause_collectives=None if steady else burst_pause_collectives,
        burst_length_collectives=None if steady else burst_length_collectives,
    )
    paper_ratio = predicted_ratio

    scenario = _scenario_key(profile, nodes, agg, steady)
    anchor_check = compare_to_anchor(profile.name, scenario, paper_ratio)

    return {
        "system": profile.name.value,
        "nodes": nodes,
        "victim": vic.value,
        "aggressor": agg.value,
        "steady": steady,
        "message_bytes": message_bytes,
        "runtime_ms_uncongested": round(t_base, 4),
        "runtime_ms_congested": round(t_cong, 4),
        "measured_ratio_uncongested_over_congested": round(measured_ratio, 4),
        "predicted_ratio_uncongested_over_congested": round(paper_ratio, 4),
        "predicted_performance_ratio": round(predicted_ratio, 4),
        "anchor": anchor_check,
    }


def _scenario_key(
    profile: FabricProfile,
    nodes: int,
    aggressor: AggressorPattern,
    steady: bool,
) -> str:
    if steady:
        return f"{profile.name.value}_{aggressor.value}_{nodes}"
    return f"{profile.name.value}_{aggressor.value}_bursty_{nodes}"


def sweep_steady_heatmap(
    system: SystemName | str,
    *,
    node_counts: list[int] | None = None,
    aggressor: AggressorPattern | str = AggressorPattern.ALLTOALL,
) -> list[dict[str, Any]]:
    """Sparse heatmap rows mimicking Fig. 5 (ratio vs nodes × message size)."""
    from ltx_trainer.hpc_interconnect.collectives import log2_message_axis

    node_counts = node_counts or [16, 32, 64, 128, 256]
    rows: list[dict[str, Any]] = []
    for n in node_counts:
        for mb in log2_message_axis()[:6]:
            pr = predict_performance_ratio(
                system,
                nodes=n,
                aggressor=aggressor,
                message_bytes=mb,
                steady=True,
            )
            rows.append({"nodes": n, "message_bytes": mb, "performance_ratio": round(pr, 3)})
    return rows
