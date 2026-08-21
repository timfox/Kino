"""BBR-Copilot config (arXiv:2606.03468)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBRCopilotConfig:
    paper_arxiv: str = "arXiv:2606.03468"
    packages: tuple[str, ...] = (
        "bandwidth",
        "bbr",
        "livestream",
        "copilot",
        "simulation",
        "metrics",
    )


@dataclass(frozen=True)
class LiveStreamConfig:
    """On-off live streaming source (§2.2)."""

    bitrate_mbps: float = 5.4
    frame_interval_s: float = 1.0 / 30.0
    gop_duration_s: float = 3.0
    gop_bytes: int | None = None  # default: bitrate * gop_duration


@dataclass(frozen=True)
class MahimahiTestbedConfig:
    """Mahimahi-style testbed defaults (§4.1)."""

    duration_s: float = 60.0
    rtt_ms: float = 200.0
    buffer_kb: int = 40
    uplink_mbps: float = 37.2
    downlink_mbps: float = 37.2
    live: LiveStreamConfig = LiveStreamConfig()


@dataclass(frozen=True)
class ProbeBWScenario:
    """§4.3 stepped bandwidth trace."""

    rtt_ms: float = 50.0
    uplink_mbps: float = 12.0
    step_high_mbps: float = 24.0
    step_at_s: tuple[float, float] = (20.0, 40.0)
    duration_s: float = 60.0
    live: LiveStreamConfig = LiveStreamConfig()
