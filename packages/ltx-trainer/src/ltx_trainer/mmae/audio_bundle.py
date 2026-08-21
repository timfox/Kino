"""Resolve MMAE rubric audio clips and encode for multimodal judgers."""

from __future__ import annotations

import base64
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from ltx_trainer.mmae.audio_refs import parse_audio_refs
from ltx_trainer.mmae.sample import MMAESample, Rubric


@dataclass(frozen=True)
class AudioClip:
    label: str
    path: Path
    start_sec: float | None = None
    end_sec: float | None = None

    @property
    def needs_slice(self) -> bool:
        return self.start_sec is not None or self.end_sec is not None


def normalize_audio_label(label: str) -> str:
    token = label.strip().lower()
    token = re.sub(r"\s+", "", token)
    if token.startswith("input"):
        return token
    if token == "output":
        return "output"
    return token


def _resolve_output_wav(predictions_dir: Path | None, sample_id: str) -> Path | None:
    if predictions_dir is None:
        return None
    from ltx_trainer.mmae.paths import resolve_prediction_wav

    return resolve_prediction_wav(predictions_dir, sample_id)


def _resolve_input_wav(dataset_root: Path | None, sample: MMAESample, index: int) -> Path | None:
    if dataset_root is None or index >= len(sample.audio_paths):
        return None
    path = dataset_root / sample.audio_paths[index]
    return path if path.is_file() else None


def resolve_clip_path(
    label: str,
    sample: MMAESample,
    *,
    dataset_root: Path | None,
    predictions_dir: Path | None,
) -> Path | None:
    norm = normalize_audio_label(label)
    if norm == "output":
        return _resolve_output_wav(predictions_dir, sample.sample_id)
    if norm.startswith("input"):
        suffix = norm.replace("input", "", 1)
        idx = int(suffix) - 1 if suffix.isdigit() else 0
        return _resolve_input_wav(dataset_root, sample, idx)
    return None


def clips_for_rubric(
    rubric: Rubric,
    sample: MMAESample,
    *,
    dataset_root: Path | None,
    predictions_dir: Path | None,
) -> list[AudioClip]:
    refs = parse_audio_refs(rubric.question)
    labels = [normalize_audio_label(r.label) for r in refs]
    if not labels:
        labels = ["output"] + [f"input{i}" for i in range(1, len(sample.audio_paths) + 1)]
    seen: set[str] = set()
    clips: list[AudioClip] = []
    ref_by_label = {normalize_audio_label(r.label): r for r in refs}
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        path = resolve_clip_path(label, sample, dataset_root=dataset_root, predictions_dir=predictions_dir)
        if path is None:
            continue
        ref = ref_by_label.get(label)
        clips.append(
            AudioClip(
                label=label,
                path=path,
                start_sec=ref.start_sec if ref else None,
                end_sec=ref.end_sec if ref else None,
            )
        )
    return clips


def _resolve_bounds(
    start_sec: float | None,
    end_sec: float | None,
    duration: float,
) -> tuple[float, float]:
    start = 0.0 if start_sec is None else float(start_sec)
    end = duration if end_sec is None else float(end_sec)
    if start < 0:
        start = max(0.0, duration + start)
    if end < 0:
        end = max(0.0, duration + end)
    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    return start, end


def slice_wav_pcm(
    path: Path,
    start_sec: float | None,
    end_sec: float | None,
    *,
    out_path: Path,
) -> Path:
    """Trim a PCM WAV using stdlib ``wave`` (Appendix C time-slice refs)."""
    with wave.open(str(path), "rb") as src:
        framerate = src.getframerate()
        nframes = src.getnframes()
        duration = nframes / framerate if framerate else 0.0
        start, end = _resolve_bounds(start_sec, end_sec, duration)
        if start <= 0.0 and end >= duration - 1e-6:
            return path
        start_frame = int(start * framerate)
        end_frame = max(start_frame, int(end * framerate))
        src.setpos(start_frame)
        frames = src.readframes(end_frame - start_frame)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as dst:
            dst.setnchannels(src.getnchannels())
            dst.setsampwidth(src.getsampwidth())
            dst.setframerate(framerate)
            dst.setcomptype(src.getcomptype(), src.getcompname())
            dst.writeframes(frames)
    return out_path


def materialize_clip(clip: AudioClip, *, cache_dir: Path | None = None) -> Path:
    """Return on-disk WAV for judger upload, applying rubric time slices when set."""
    if not clip.needs_slice or not clip.path.is_file():
        return clip.path
    try:
        with wave.open(str(clip.path), "rb") as probe:
            framerate = probe.getframerate()
            nframes = probe.getnframes()
            duration = nframes / framerate if framerate else 0.0
            start, end = _resolve_bounds(clip.start_sec, clip.end_sec, duration)
            if start <= 0.0 and end >= duration - 1e-6:
                return clip.path
    except (wave.Error, OSError, ValueError):
        return clip.path
    root = cache_dir or Path(tempfile.gettempdir()) / "gopex_mmae_slices"
    root.mkdir(parents=True, exist_ok=True)
    stamp = clip.path.stat().st_mtime_ns
    out = root / (
        f"{clip.path.stem}_{clip.label}_{clip.start_sec}_{clip.end_sec}_{stamp}.slice.wav"
    )
    if out.is_file():
        return out
    try:
        return slice_wav_pcm(clip.path, clip.start_sec, clip.end_sec, out_path=out)
    except (wave.Error, OSError, ValueError):
        return clip.path


def wav_to_data_url(path: Path) -> str:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{data}"


def audio_part(path: Path, *, part_style: str = "input_audio") -> dict:
    data_url = wav_to_data_url(path)
    if part_style == "audio_url":
        return {"type": "audio_url", "audio_url": {"url": data_url}}
    return {"type": "input_audio", "input_audio": {"data": data_url.split(",", 1)[1], "format": "wav"}}


def build_multimodal_user_content(
    rubric: Rubric,
    choices: list[str],
    clips: list[AudioClip],
    *,
    part_style: str = "input_audio",
    cache_dir: Path | None = None,
) -> list[dict]:
    from ltx_trainer.mmae.judger import format_user_prompt

    header_lines = [f"<audio {clip.label}>: attached below" for clip in clips]
    prompt = format_user_prompt(rubric.question, choices, audio_labels=[c.label for c in clips])
    if header_lines:
        prompt = "\n".join(header_lines) + "\n\n" + prompt.split("\n\n", 1)[-1]
    content: list[dict] = [{"type": "text", "text": prompt}]
    for clip in clips:
        path = materialize_clip(clip, cache_dir=cache_dir)
        content.append(audio_part(path, part_style=part_style))
    return content
