"""Importance-weighted frame sampling for vLLM vision captioning."""

from __future__ import annotations

import math
from pathlib import Path

from ltx_trainer.dyna_pruner.env import dyna_pruner_enabled
from ltx_trainer.ffmpeg_io import duration_seconds, extract_frame_jpeg, sample_frame_times


def _score_jpeg_motion(jpeg_paths: list[Path]) -> list[float]:
    scores: list[float] = []
    prev: bytes | None = None
    for jp in jpeg_paths:
        raw = jp.read_bytes()
        if prev is None:
            scores.append(0.0)
        else:
            n = min(len(raw), len(prev), 8192)
            diff = sum(abs(a - b) for a, b in zip(raw[:n], prev[:n], strict=False))
            scores.append(float(diff) / max(n, 1))
        prev = raw
    return scores


def importance_sample_frame_times(
    video: Path,
    duration: float | None,
    n: int,
    *,
    candidates: int | None = None,
) -> list[float]:
    """
    Sample ``n`` timestamps with highest motion/importance for vLLM captioning.

    Uses coarse JPEG extracts (ffmpeg) — reduces vLLM prefill tokens vs uniform sampling
    when ``GOPEX_DYNA_PRUNER=1``.
    """
    n = max(1, int(n))
    if not dyna_pruner_enabled():
        return sample_frame_times(duration, n)
    cand_n = max(n, int(candidates or n * 4))
    times = sample_frame_times(duration, cand_n)
    if len(times) <= n:
        return times
    import tempfile

    scored: list[tuple[float, float]] = []
    with tempfile.TemporaryDirectory(prefix="dyna_vlm_") as td:
        tdir = Path(td)
        for i, t in enumerate(times):
            jp = tdir / f"c_{i}.jpg"
            if not extract_frame_jpeg(video, t, jp):
                continue
            scored.append((t, 0.0))
        if len(scored) < n:
            return times[:n]
        paths = [tdir / f"c_{i}.jpg" for i in range(len(scored))]
        motion = _score_jpeg_motion([p for p in paths if p.is_file()])
        for idx, m in enumerate(motion):
            scored[idx] = (scored[idx][0], m)
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = sorted(t for t, _ in scored[:n])
    return picked


def resolve_frame_times(
    video: Path,
    duration: float | None,
    n: int,
) -> list[float]:
    dur = duration
    if dur is None:
        try:
            dur = duration_seconds(video)
        except Exception:
            dur = None
    if dyna_pruner_enabled():
        return importance_sample_frame_times(video, dur, n)
    return sample_frame_times(dur, n)
