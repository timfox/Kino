"""Segment-Aware 3D RoPE (SA-3D RoPE) — Sec. 2.1.2, Eq. (2)."""

from __future__ import annotations

import math


def base_spatiotemporal_phase(t: int, h: int, w: int, *, base_scale: float = 0.01) -> float:
    """Scalar toy for 3D RoPE phase before segment modulation."""
    return base_scale * (t + h * 1.7 + w * 2.3)


def segment_phase_offset(segment_index: int, *, dim_per_segment: float = 0.35) -> float:
    """r_seg(i) — extra phase per segment index (toy)."""
    return dim_per_segment * segment_index


def sa_3d_rope_modulated_phase(t: int, h: int, w: int, segment_index: int) -> float:
    r"""˜r_{t,h,w,i} = r_{t,h,w} · r_seg(i) — additive phase stub (Eq. 2 narrative)."""
    return base_spatiotemporal_phase(t, h, w) + segment_phase_offset(segment_index)
