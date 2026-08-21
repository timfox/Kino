"""Steady and bursty congestion injection profiles (Sec. III-C–D)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.hpc_interconnect.fabrics import AggressorPattern, FabricProfile


@dataclass(frozen=True)
class SteadyCongestion:
    aggressor: AggressorPattern = AggressorPattern.ALLTOALL

    def slowdown_multiplier(self, profile: FabricProfile, *, nodes: int) -> float:
        return 1.0 / max(1e-3, _steady_ratio(profile, nodes, self.aggressor))


@dataclass(frozen=True)
class BurstyCongestion:
    aggressor: AggressorPattern = AggressorPattern.ALLTOALL
    burst_collectives: int = 4
    pause_collectives: int = 2  # idle gap between bursts (in aggressor iterations)

    def duty_cycle(self) -> float:
        return self.burst_collectives / max(1, self.burst_collectives + self.pause_collectives)

    def slowdown_multiplier(self, profile: FabricProfile, *, nodes: int) -> float:
        base = _steady_ratio(profile, nodes, self.aggressor)
        duty = self.duty_cycle()
        gap_factor = min(1.0, self.pause_collectives / 4.0)
        recovery = profile.bursty_recovery * gap_factor
        effective = base + (1.0 - base) * (1.0 - recovery) * duty
        if self.aggressor == AggressorPattern.INCAST and self.pause_collectives <= 1:
            effective *= 0.55
        return 1.0 / max(1e-3, effective)


def _steady_ratio(profile: FabricProfile, nodes: int, aggressor: AggressorPattern) -> float:
    from ltx_trainer.hpc_interconnect.simulator import predict_performance_ratio

    return predict_performance_ratio(
        profile.name,
        nodes=nodes,
        aggressor=aggressor,
        message_bytes=32 * 1024,
        steady=True,
    )
