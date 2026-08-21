"""Event-aligned segmentation for long-form evaluation (Sec. 4.2)."""

from __future__ import annotations

import re
from typing import Any

from ltx_trainer.longav_compass.annotation import BenchmarkCase, EventAnnotation

_TIME_RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*s?\s*$",
    re.IGNORECASE,
)


def parse_time_range(spec: str) -> tuple[float, float]:
    """Parse strings like ``0-18s`` or ``8–13s`` into ``(start_s, end_s)``."""
    m = _TIME_RANGE_RE.match(spec.strip())
    if not m:
        raise ValueError(f"invalid time range: {spec!r}")
    return float(m.group(1)), float(m.group(2))


def event_duration_s(event: EventAnnotation) -> float:
    start, end = parse_time_range(event.time_range)
    return max(0.0, end - start)


def event_segment_specs(case: BenchmarkCase) -> list[dict[str, Any]]:
    """Canonical event-aligned clip windows for per-event metrics."""
    rows: list[dict[str, Any]] = []
    for ev in case.events:
        start, end = parse_time_range(ev.time_range)
        rows.append(
            {
                "event_id": ev.event_id,
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "action": ev.action,
                "clip_path": f"{case.case_id}/events/event_{ev.event_id:02d}.mp4",
            }
        )
    return rows


def boundary_clip_windows(
    case: BenchmarkCase,
    *,
    pad_s: float = 2.0,
) -> list[dict[str, Any]]:
    """2 s before/after each event boundary for transition-stability (Trans.)."""
    if len(case.events) < 2:
        return []
    ordered = sorted(case.events, key=lambda e: e.event_id)
    windows: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        _, left_end = parse_time_range(left.time_range)
        right_start, _ = parse_time_range(right.time_range)
        center = (left_end + right_start) / 2.0
        windows.append(
            {
                "between_events": (left.event_id, right.event_id),
                "center_s": center,
                "start_s": max(0.0, center - pad_s),
                "end_s": center + pad_s,
                "clip_path": (
                    f"{case.case_id}/boundaries/boundary_{left.event_id:02d}_{right.event_id:02d}.mp4"
                ),
            }
        )
    return windows


def ffmpeg_segment_commands(
    case: BenchmarkCase,
    *,
    input_video: str = "full_video.mp4",
) -> list[dict[str, str]]:
    """FFmpeg command templates for event and boundary clips (Appendix D.3)."""
    cmds: list[dict[str, str]] = []
    for seg in event_segment_specs(case):
        out = seg["clip_path"]
        cmds.append(
            {
                "kind": "event",
                "event_id": str(seg["event_id"]),
                "command": (
                    f"ffmpeg -y -ss {seg['start_s']:.3f} -to {seg['end_s']:.3f} "
                    f"-i {input_video} -c copy {out}"
                ),
            }
        )
    for win in boundary_clip_windows(case):
        out = win["clip_path"]
        cmds.append(
            {
                "kind": "boundary",
                "between": f"{win['between_events'][0]}_{win['between_events'][1]}",
                "command": (
                    f"ffmpeg -y -ss {win['start_s']:.3f} -to {win['end_s']:.3f} "
                    f"-i {input_video} -c copy {out}"
                ),
            }
        )
    return cmds


def canonical_events_json(case: BenchmarkCase) -> dict[str, Any]:
    """Layout expected under each model output directory (Appendix D.3)."""
    return {
        "case_id": case.case_id,
        "task": case.task,
        "full_video": f"{case.case_id}/full_video.mp4",
        "events": event_segment_specs(case),
        "boundaries": boundary_clip_windows(case),
    }
