"""TAL helpers: ShuttleSet interval conversion and proposal matching."""

from __future__ import annotations


def shuttleset_stroke_interval(
    frame_t: int,
    *,
    half_window: int = 9,
    max_frame: int | None = None,
) -> tuple[int, int]:
    """Convert stroke-point at t to [t-9, t+9] (19 frames) per Sec. V-A."""
    start = max(0, frame_t - half_window)
    end = frame_t + half_window
    if max_frame is not None:
        end = min(max_frame, end)
    return start, end


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (end_a - start_a) + (end_b - start_b) - inter
    if union <= 0:
        return 0.0
    return inter / union
