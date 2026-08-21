"""Live-stream on-off traffic (§2.2, Fig. 2)."""

from __future__ import annotations

from ltx_trainer.bbr_copilot.config import LiveStreamConfig


def gop_bytes(cfg: LiveStreamConfig) -> int:
    if cfg.gop_bytes is not None:
        return cfg.gop_bytes
    return int(cfg.bitrate_mbps * 1e6 / 8.0 * cfg.gop_duration_s)


def frame_bytes(cfg: LiveStreamConfig) -> int:
    return int(cfg.bitrate_mbps * 1e6 / 8.0 * cfg.frame_interval_s)


def app_bytes_available(t: float, cfg: LiveStreamConfig) -> float:
    """Bytes ready in send buffer at time t (GOP burst then periodic frames)."""
    if t < cfg.gop_duration_s:
        rate = gop_bytes(cfg) / cfg.gop_duration_s
        return rate * cfg.frame_interval_s
    # one frame worth per frame interval
    return float(frame_bytes(cfg))
