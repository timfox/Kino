"""Batch prompt / shot-list support for full composed movie renders."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ltx_trainer.ltx3.frames import duration_for_frames, frames_for_duration, legal_frame_count
from ltx_trainer.ltx3.minute_compose import MinuteComposePlan, SegmentSpec
from ltx_trainer.ltx3.render_plan import MinuteRenderPlan, build_minute_render_plan

MOVIE_SPEECH_NEGATIVE = (
    "unintelligible speech, gibberish speech, pseudo-English, fake words, mumbling, "
    "garbled dialogue, nonsensical language, lip-sync dialogue, dubbed dialogue, "
    "subtitles, captions, on-screen text"
)


@dataclass
class MovieShot:
    shot_id: str
    scene: str
    prompt: str
    duration_s: float = 5.0
    negative_prompt: str = ""
    notes: str = ""
    tags: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean(s: Any) -> str:
    return str(s or "").strip()


def _duration_from_value(raw: Any, *, fps: float, default_s: float) -> float:
    text = _clean(raw)
    if not text:
        return default_s
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*s(?:ec(?:onds?)?)?\b", text, re.I)
    if m:
        return max(0.1, float(m.group(1)))
    try:
        val = float(text)
    except ValueError:
        return default_s
    if val > 30:
        return max(0.1, val / max(fps, 1e-6))
    return max(0.1, val)


def _shot_prompt(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


def _parse_inline_duration(line: str, *, default_s: float) -> tuple[float, str]:
    s = line.strip()
    m = re.match(r"^\[(\d+(?:\.\d+)?)\s*s\]\s*(.*)$", s, re.I)
    if m:
        return max(0.1, float(m.group(1))), m.group(2).strip()
    m = re.match(r"^(?:duration|dur|seconds|sec)\s*[:=]\s*(\d+(?:\.\d+)?)\s*s?\s*\|\s*(.*)$", s, re.I)
    if m:
        return max(0.1, float(m.group(1))), m.group(2).strip()
    return default_s, s


def parse_text_prompt_batch(path: str | Path, *, default_duration_s: float = 5.0) -> list[MovieShot]:
    """Parse a simple Markdown/text batch into shots.

    Supported lines:
      ``# Scene Name``
      ``- [5s] A shot prompt``
      ``duration=8 | A shot prompt``
      blank-line separated paragraphs
    """
    p = Path(path).expanduser().resolve()
    scene = "Scene 1"
    shots: list[MovieShot] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(paragraph).strip()
        paragraph = []
        if not text:
            return
        duration, prompt = _parse_inline_duration(text, default_s=default_duration_s)
        shots.append(MovieShot(shot_id=f"{len(shots) + 1:03d}", scene=scene, prompt=prompt, duration_s=duration))

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("#"):
            flush_paragraph()
            scene = line.lstrip("#").strip() or scene
            continue
        bullet = re.match(r"^(?:[-*]|\d+[\).]|shot\s+\d+\s*[:.-])\s*(.*)$", line, re.I)
        if bullet:
            flush_paragraph()
            duration, prompt = _parse_inline_duration(bullet.group(1).strip(), default_s=default_duration_s)
            shots.append(MovieShot(shot_id=f"{len(shots) + 1:03d}", scene=scene, prompt=prompt, duration_s=duration))
            continue
        paragraph.append(line)
    flush_paragraph()
    return [s for s in shots if s.prompt]


def parse_csv_prompt_batch(path: str | Path, *, fps: float = 24.0, default_duration_s: float = 5.0) -> list[MovieShot]:
    p = Path(path).expanduser().resolve()
    with p.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    shots: list[MovieShot] = []
    for i, row in enumerate(rows, start=1):
        norm = {re.sub(r"\W+", "_", k.strip().lower()): v for k, v in row.items() if k}

        def pick(*names: str) -> str:
            for name in names:
                key = re.sub(r"\W+", "_", name.strip().lower())
                if key in norm and _clean(norm[key]):
                    return _clean(norm[key])
            return ""

        prompt = pick("prompt", "description", "shot", "shot_prompt")
        name = pick("shot_name", "name", "title")
        if not prompt:
            prompt = name
        if not prompt:
            continue
        dur_raw = pick("duration_s", "seconds", "duration", "duration_frames", "frames")
        shots.append(
            MovieShot(
                shot_id=pick("shot_id", "id") or f"{i:03d}",
                scene=pick("scene_name", "scene", "slugline") or "Scene 1",
                prompt=_shot_prompt(name, prompt) if name and name != prompt else prompt,
                duration_s=_duration_from_value(dur_raw, fps=fps, default_s=default_duration_s),
                negative_prompt=pick("negative_prompt"),
                notes=pick("notes"),
                tags=pick("tags"),
                metadata={"row": i},
            )
        )
    return shots


def parse_json_prompt_batch(path: str | Path, *, fps: float = 24.0, default_duration_s: float = 5.0) -> list[MovieShot]:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    shots: list[MovieShot] = []

    if isinstance(data, list):
        iterable = [{"name": "Movie", "shots": data}]
    else:
        iterable = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(iterable, list):
        raise ValueError("JSON must be a shot array or a Gopex shot-list object with scenes[].")

    for scene_idx, scene in enumerate(iterable, start=1):
        if not isinstance(scene, dict):
            continue
        scene_name = _clean(scene.get("name") or scene.get("scene") or f"Scene {scene_idx}")
        scene_synopsis = _clean(scene.get("synopsis") or scene.get("slugline"))
        raw_shots = scene.get("shots")
        if not isinstance(raw_shots, list):
            continue
        for shot in raw_shots:
            if not isinstance(shot, dict):
                continue
            prompt = _clean(shot.get("prompt") or shot.get("description") or shot.get("name"))
            if not prompt:
                continue
            dur_raw = shot.get("duration_s")
            if dur_raw is None:
                dur_raw = shot.get("seconds")
            if dur_raw is None:
                dur_raw = shot.get("duration_frames")
            shots.append(
                MovieShot(
                    shot_id=_clean(shot.get("shot_id") or shot.get("id") or f"{len(shots) + 1:03d}"),
                    scene=scene_name,
                    prompt=_shot_prompt(scene_synopsis, _clean(shot.get("name")), prompt),
                    duration_s=_duration_from_value(dur_raw, fps=fps, default_s=default_duration_s),
                    negative_prompt=_clean(shot.get("negative_prompt")),
                    notes=_clean(shot.get("notes")),
                    tags=_clean(shot.get("tags")),
                    metadata={"scene_index": scene_idx},
                )
            )
    return shots


def parse_movie_prompt_batch(
    path: str | Path,
    *,
    fps: float = 24.0,
    default_duration_s: float = 5.0,
) -> list[MovieShot]:
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()
    if suffix == ".json":
        shots = parse_json_prompt_batch(p, fps=fps, default_duration_s=default_duration_s)
    elif suffix == ".csv":
        shots = parse_csv_prompt_batch(p, fps=fps, default_duration_s=default_duration_s)
    else:
        shots = parse_text_prompt_batch(p, default_duration_s=default_duration_s)
    if not shots:
        raise ValueError(f"No usable shots found in {p}")
    return shots


def compose_movie_batch_plan(
    shots: list[MovieShot],
    *,
    fps: float = 24.0,
    hard_cap_frames: int = 121,
    max_shots: int = 0,
    global_prefix: str = "",
    carry_hero: bool = True,
) -> MinuteComposePlan:
    selected = shots[:max_shots] if max_shots and max_shots > 0 else shots
    segments: list[SegmentSpec] = []
    t = 0.0
    for shot in selected:
        remaining = max(0.1, shot.duration_s)
        part = 1
        max_dur = duration_for_frames(frames=legal_frame_count(hard_cap_frames), fps=fps)
        total_frames = frames_for_duration(duration_s=remaining, fps=fps)
        while remaining > 1e-6:
            if part == 1 and total_frames <= hard_cap_frames:
                frames = total_frames
            else:
                chunk_s = min(remaining, max_dur)
                frames = frames_for_duration(duration_s=chunk_s, fps=fps)
                frames = legal_frame_count(min(frames, hard_cap_frames))
                if frames <= 1 and part > 1:
                    break
            dur = duration_for_frames(frames=frames, fps=fps)
            prompt_bits = [
                global_prefix,
                f"Scene: {shot.scene}.",
                f"Shot {shot.shot_id}" + (f" part {part}." if shot.duration_s > max_dur + 1e-6 else "."),
                shot.prompt,
            ]
            if shot.tags:
                prompt_bits.append(f"Tags: {shot.tags}.")
            if shot.notes:
                prompt_bits.append(f"Notes: {shot.notes}.")
            negative = _shot_prompt(MOVIE_SPEECH_NEGATIVE, shot.negative_prompt)
            segments.append(
                SegmentSpec(
                    index=len(segments),
                    start_s=t,
                    end_s=t + dur,
                    frames=frames,
                    fps=fps,
                    prompt=_shot_prompt(*prompt_bits),
                    negative_prompt=negative,
                    carry_hero=carry_hero and len(segments) > 0,
                    notes=[f"scene={shot.scene}", f"shot={shot.shot_id}", f"part={part}"],
                )
            )
            t += dur
            remaining -= dur
            part += 1
    return MinuteComposePlan(
        target_duration_s=t,
        fps=fps,
        segments=segments,
        strategy="movie_batch_shotlist",
        metadata={
            "shots": len(selected),
            "source_shots": len(shots),
            "hard_cap_frames": hard_cap_frames,
            "global_prefix": global_prefix,
        },
    )


def render_plan_from_prompt_batch(
    *,
    prompt_file: str | Path,
    out_dir: str | Path,
    run_name: str = "movie",
    fps: float = 24.0,
    default_duration_s: float = 5.0,
    hard_cap_frames: int = 121,
    max_shots: int = 0,
    global_prefix: str = "",
    env: Any = None,
) -> MinuteRenderPlan:
    shots = parse_movie_prompt_batch(prompt_file, fps=fps, default_duration_s=default_duration_s)
    compose = compose_movie_batch_plan(
        shots,
        fps=fps,
        hard_cap_frames=hard_cap_frames,
        max_shots=max_shots,
        global_prefix=global_prefix,
    )
    return build_minute_render_plan(compose, out_dir=out_dir, env=env, run_name=run_name)


def movie_template_text() -> str:
    return """# My Movie

## Scene 1 - Opening
- [5s] Wide establishing shot of the rain-soaked street, cinematic lighting, slow dolly forward.
- duration=6 | Close-up of the lead character listening at a half-open door, shallow depth of field.

## Scene 2 - The Discovery
- [5s] The camera tracks through a candlelit room toward an object on the table.
- [7s] The object moves by itself as the character steps back, restrained horror performance, no jump scare.
"""
