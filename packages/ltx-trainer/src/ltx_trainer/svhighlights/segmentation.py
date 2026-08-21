"""Context-aware segmentation: shot boundaries + transcript merge (Stage 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ltx_trainer.svhighlights.config import SvHighlightsConfig


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    shot_ids: tuple[int, ...]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class WordSpan:
    start_s: float
    end_s: float
    text: str


def detect_shots_stub(
    duration_s: float,
    *,
    fps: float = 30.0,
    mean_shot_s: float = 4.0,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Synthetic shot boundaries (TransNet V2 proxy)."""
    rng = np.random.default_rng(seed)
    shots: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_s - 0.5:
        length = float(rng.uniform(mean_shot_s * 0.5, mean_shot_s * 1.5))
        end = min(t + length, duration_s)
        shots.append((t, end))
        t = end
    if not shots:
        shots.append((0.0, duration_s))
    return shots


def sentence_spans_from_words(words: Sequence[WordSpan], *, gap_s: float = 1.0) -> list[tuple[float, float]]:
    if not words:
        return []
    spans: list[tuple[float, float]] = []
    start = words[0].start_s
    prev_end = words[0].end_s
    for w in words[1:]:
        if w.start_s - prev_end >= gap_s:
            spans.append((start, prev_end))
            start = w.start_s
        prev_end = w.end_s
    spans.append((start, prev_end))
    return spans


def merge_shots_by_transcript(
    shots: Sequence[tuple[float, float]],
    words: Sequence[WordSpan],
    *,
    max_segment_s: float = 120.0,
    gap_s: float = 1.0,
) -> list[Segment]:
    """Merge adjacent shots when a transcript sentence spans both."""
    if not shots:
        return []
    sentences = sentence_spans_from_words(words, gap_s=gap_s)
    merged: list[Segment] = []
    i = 0
    while i < len(shots):
        start, end = shots[i]
        shot_ids = [i]
        j = i + 1
        while j < len(shots):
            candidate_end = shots[j][1]
            if candidate_end - start > max_segment_s:
                break
            spans_both = any(
                s_start <= shots[j - 1][1] and s_end >= shots[j][0]
                for s_start, s_end in sentences
            )
            if not spans_both:
                break
            end = candidate_end
            shot_ids.append(j)
            j += 1
        merged.append(Segment(start_s=start, end_s=end, shot_ids=tuple(shot_ids)))
        i = j if j > i + 1 else i + 1
    return merged


def segment_video_stub(
    duration_s: float,
    words: Sequence[WordSpan] | None = None,
    cfg: SvHighlightsConfig | None = None,
    *,
    seed: int = 0,
) -> list[Segment]:
    from ltx_trainer.svhighlights.config import SvHighlightsConfig

    c = cfg or SvHighlightsConfig()
    shots = detect_shots_stub(duration_s, seed=seed)
    if words is None:
        rng = np.random.default_rng(seed)
        words = []
        t = 0.0
        while t < duration_s - 1.0:
            length = float(rng.uniform(0.2, 0.6))
            words.append(WordSpan(t, t + length, "play"))
            t += float(rng.uniform(0.5, 2.5))
    return merge_shots_by_transcript(
        shots,
        words,
        max_segment_s=c.max_segment_length_s,
        gap_s=c.transcript_gap_merge_s,
    )
