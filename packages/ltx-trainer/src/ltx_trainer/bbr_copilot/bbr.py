"""Simplified BBRv1 state machine (§2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BBRPhase(str, Enum):
    STARTUP = "startup"
    DRAIN = "drain"
    PROBE_BW = "probe_bw"
    PROBE_RTT = "probe_rtt"


STARTUP_GAIN = 2.885
PROBE_GAIN = 1.25
STARTUP_EXIT_ROUNDS = 3
STARTUP_GROWTH_RATIO = 1.25


@dataclass
class BBRState:
    phase: BBRPhase = BBRPhase.STARTUP
    btl_bw: float = 1e6  # bytes/s initial guess
    rt_prop: float = 0.2
    pacing_gain: float = STARTUP_GAIN
    startup_stall_rounds: int = 0
    probe_cycle: int = 0
    max_filter_window: list[float] = field(default_factory=list)

    def bdp(self) -> float:
        return self.btl_bw * self.rt_prop

    def pacing_rate(self) -> float:
        return self.btl_bw * self.pacing_gain

    def update_bandwidth_sample(self, sample: float | None, *, last_sample_accurate: bool) -> None:
        if sample is None or sample <= 0.0:
            return
        self.max_filter_window.append(sample)
        if len(self.max_filter_window) > 10:
            self.max_filter_window.pop(0)
        new_bw = max(self.max_filter_window)
        if self.phase == BBRPhase.STARTUP:
            if not last_sample_accurate:
                return
            if new_bw >= self.btl_bw * STARTUP_GROWTH_RATIO:
                self.btl_bw = new_bw
                self.startup_stall_rounds = 0
            else:
                self.startup_stall_rounds += 1
                self.btl_bw = max(self.btl_bw, new_bw)
            if self.startup_stall_rounds >= STARTUP_EXIT_ROUNDS:
                self.phase = BBRPhase.DRAIN
                self.pacing_gain = 1.0 / STARTUP_GAIN
        elif self.phase == BBRPhase.PROBE_BW:
            if sample > self.btl_bw:
                self.btl_bw = sample

    def advance_phase(self, *, inflight: float) -> None:
        if self.phase == BBRPhase.DRAIN and inflight <= self.bdp():
            self.phase = BBRPhase.PROBE_BW
            self.pacing_gain = 1.0
            self.probe_cycle = 0
        elif self.phase == BBRPhase.PROBE_BW:
            self.probe_cycle = (self.probe_cycle + 1) % 8
            self.pacing_gain = PROBE_GAIN if self.probe_cycle == 0 else 1.0
        elif self.phase == BBRPhase.STARTUP:
            self.pacing_gain = STARTUP_GAIN
