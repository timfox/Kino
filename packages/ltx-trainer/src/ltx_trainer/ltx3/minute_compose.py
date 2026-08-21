"""Minute-scale AV generation plans for LTX 2.3 (composed segments, 3.0-style product path)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.ltx3.frames import duration_for_frames, frames_for_duration, legal_frame_count
from ltx_trainer.longav_compass.annotation import BenchmarkCase, build_generation_prompt
from ltx_trainer.longav_compass.segmentation import parse_time_range


@dataclass
class SegmentSpec:
    index: int
    start_s: float
    end_s: float
    frames: int
    fps: float
    prompt: str
    negative_prompt: str = ""
    pipeline: str = "ti2vid_two_stages_hq"
    carry_hero: bool = False
    retake_fallback: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class MinuteComposePlan:
    target_duration_s: float
    fps: float
    segments: list[SegmentSpec]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_planned_s(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end_s - self.segments[0].start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_duration_s": self.target_duration_s,
            "fps": self.fps,
            "strategy": self.strategy,
            "total_planned_s": round(self.total_planned_s, 3),
            "metadata": self.metadata,
            "segments": [
                {
                    "index": s.index,
                    "start_s": round(s.start_s, 3),
                    "end_s": round(s.end_s, 3),
                    "frames": s.frames,
                    "duration_s": round(s.duration_s, 3),
                    "pipeline": s.pipeline,
                    "carry_hero": s.carry_hero,
                    "retake_fallback": s.retake_fallback,
                    "prompt_preview": s.prompt[:240] + ("…" if len(s.prompt) > 240 else ""),
                    "negative_prompt": s.negative_prompt,
                    "notes": s.notes,
                }
                for s in self.segments
            ],
        }


def _max_segment_frames(*, max_segment_s: float, fps: float, hard_cap_frames: int) -> int:
    want = frames_for_duration(duration_s=max_segment_s, fps=fps)
    return legal_frame_count(min(want, hard_cap_frames))


def compose_minute_plan(
    *,
    global_prompt: str,
    target_duration_s: float = 60.0,
    fps: float = 24.0,
    max_segment_s: float = 5.0,
    hard_cap_frames: int = 121,
    overlap_s: float = 0.0,
) -> MinuteComposePlan:
    """
    Uniform segment decomposition when no event annotations exist.

    Each segment gets the same global prompt (Gemma long-context shot list). Segment 1+
    should use Consistency hero from the last frame of the previous segment (``carry_hero``).
    """
    seg_frames = _max_segment_frames(
        max_segment_s=max_segment_s, fps=fps, hard_cap_frames=hard_cap_frames
    )
    seg_dur = duration_for_frames(frames=seg_frames, fps=fps)
    step = max(seg_dur - max(0.0, overlap_s), seg_dur * 0.5)

    segments: list[SegmentSpec] = []
    t = 0.0
    idx = 0
    while t < target_duration_s - 1e-6:
        remaining = target_duration_s - t
        if remaining <= seg_dur + 1e-6:
            frames = frames_for_duration(duration_s=remaining, fps=fps)
            frames = legal_frame_count(min(frames, hard_cap_frames))
            end_s = min(target_duration_s, t + duration_for_frames(frames=frames, fps=fps))
        else:
            frames = seg_frames
            end_s = t + duration_for_frames(frames=frames, fps=fps)
        segments.append(
            SegmentSpec(
                index=idx,
                start_s=t,
                end_s=end_s,
                frames=frames,
                fps=fps,
                prompt=global_prompt.strip(),
                carry_hero=idx > 0,
                notes=["uniform_decomposition"],
            )
        )
        idx += 1
        if end_s >= target_duration_s - 1e-6:
            break
        t += step

    return MinuteComposePlan(
        target_duration_s=target_duration_s,
        fps=fps,
        segments=segments,
        strategy="uniform",
        metadata={
            "max_segment_s": max_segment_s,
            "hard_cap_frames": hard_cap_frames,
            "overlap_s": overlap_s,
        },
    )


def segment_from_events(
    case: BenchmarkCase,
    *,
    fps: float = 24.0,
    hard_cap_frames: int = 121,
    split_long_events: bool = True,
) -> MinuteComposePlan:
    """Event-aligned segments for LongAV-Compass cases (preserve event order)."""
    prompts = build_generation_prompt(case)
    global_prompt = prompts.get("video_prompt") or case.global_description

    segments: list[SegmentSpec] = []
    idx = 0
    for ev in sorted(case.events, key=lambda e: e.event_id):
        start, end = parse_time_range(ev.time_range)
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        max_dur = duration_for_frames(frames=hard_cap_frames, fps=fps)
        if split_long_events and dur > max_dur + 1e-6:
            sub_start = start
            while sub_start < end - 1e-6:
                sub_end = min(end, sub_start + max_dur)
                frames = frames_for_duration(duration_s=sub_end - sub_start, fps=fps)
                frames = legal_frame_count(min(frames, hard_cap_frames))
                sub_prompt = f"{global_prompt}\n\n[Event {ev.event_id}] {ev.action}"
                segments.append(
                    SegmentSpec(
                        index=idx,
                        start_s=sub_start,
                        end_s=sub_start + duration_for_frames(frames=frames, fps=fps),
                        frames=frames,
                        fps=fps,
                        prompt=sub_prompt.strip(),
                        carry_hero=idx > 0,
                        notes=[f"event_{ev.event_id}_part"],
                    )
                )
                idx += 1
                sub_start = segments[-1].end_s
        else:
            frames = frames_for_duration(duration_s=dur, fps=fps)
            frames = legal_frame_count(min(frames, hard_cap_frames))
            sub_prompt = f"{global_prompt}\n\n[Event {ev.event_id}] {ev.action}"
            segments.append(
                SegmentSpec(
                    index=idx,
                    start_s=start,
                    end_s=start + duration_for_frames(frames=frames, fps=fps),
                    frames=frames,
                    fps=fps,
                    prompt=sub_prompt.strip(),
                    carry_hero=idx > 0,
                    notes=[f"event_{ev.event_id}"],
                )
            )
            idx += 1

    if segments:
        _, last_event_end = parse_time_range(case.events[-1].time_range)
        target = max(segments[-1].end_s, last_event_end)
    elif case.events:
        _, target = parse_time_range(case.events[-1].time_range)
    else:
        target = 60.0

    return MinuteComposePlan(
        target_duration_s=target,
        fps=fps,
        segments=segments,
        strategy="longav_events",
        metadata={"case_id": case.case_id, "task": case.task},
    )
