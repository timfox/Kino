"""TF-SELECTOR: segment captioning + LLM saliency + clip assignment (Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ltx_trainer.svhighlights.config import SvHighlightsConfig
from ltx_trainer.svhighlights.segmentation import Segment, WordSpan, segment_video_stub


@dataclass
class SegmentFeatures:
    segment: Segment
    caption: str
    transcript: str
    mean_volume: float
    saliency: float


_HIGHLIGHT_CUES = (
    "goal",
    "score",
    "touchdown",
    "win",
    "celebration",
    "crowd",
    "exciting",
    "fast break",
    "home run",
    "checkered flag",
)


def words_in_interval(words: Sequence[WordSpan], start_s: float, end_s: float) -> list[WordSpan]:
    return [w for w in words if w.end_s > start_s and w.start_s < end_s]


def caption_segment_stub(segment: Segment, words: Sequence[WordSpan], *, seed: int = 0) -> str:
    """InternVL2.5 proxy — template caption from transcript."""
    rng = np.random.default_rng(seed + int(segment.start_s * 10))
    local = words_in_interval(words, segment.start_s, segment.end_s)
    text = " ".join(w.text for w in local) or "game action"
    sport_phrase = rng.choice(["both teams", "players", "the crowd"])
    return (
        f"The video captures {sport_phrase} during play; transcript mentions {text}. "
        f"Segment spans {segment.start_s:.1f}s to {segment.end_s:.1f}s."
    )


def volume_stub(segment: Segment, *, seed: int = 0) -> float:
    rng = np.random.default_rng(seed + int(segment.end_s))
    base = 0.35 + 0.1 * np.sin(segment.start_s * 0.01)
    return float(np.clip(base + rng.uniform(-0.1, 0.4), 0.0, 1.0))


def llm_saliency_stub(
    caption: str,
    transcript: str,
    volume: float,
    cfg: SvHighlightsConfig | None = None,
) -> float:
    """Llama-3 proxy: keyword + volume heuristic on 0–5 scale."""
    c = cfg or SvHighlightsConfig()
    text = (caption + " " + transcript).lower()
    cue_hits = sum(1 for k in _HIGHLIGHT_CUES if k in text)
    score = 1.5 + 0.6 * cue_hits + 1.2 * volume
    if "quiet" in text or "timeout" in text:
        score -= 1.0
    return float(np.clip(score, c.saliency_min, c.saliency_max))


def build_segment_features(
    segments: Sequence[Segment],
    words: Sequence[WordSpan],
    *,
    seed: int = 0,
    cfg: SvHighlightsConfig | None = None,
) -> list[SegmentFeatures]:
    c = cfg or SvHighlightsConfig()
    out: list[SegmentFeatures] = []
    for i, seg in enumerate(segments):
        local = words_in_interval(words, seg.start_s, seg.end_s)
        transcript = " ".join(w.text for w in local)
        cap = caption_segment_stub(seg, words, seed=seed + i)
        vol = volume_stub(seg, seed=seed + i)
        sal = llm_saliency_stub(cap, transcript, vol, cfg=c)
        out.append(SegmentFeatures(segment=seg, caption=cap, transcript=transcript, mean_volume=vol, saliency=sal))
    return out


def clip_scores_from_segments(
    duration_s: float,
    features: Sequence[SegmentFeatures],
    *,
    clip_duration_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 2 — weighted overlap of segment saliency onto fixed 2 s clips."""
    n_clips = max(int(np.ceil(duration_s / clip_duration_s)), 1)
    scores = np.zeros(n_clips, dtype=np.float64)
    starts = np.arange(n_clips, dtype=np.float64) * clip_duration_s
    for i, cs in enumerate(starts):
        ce = min(cs + clip_duration_s, duration_s)
        clip_len = ce - cs
        if clip_len <= 0:
            continue
        total = 0.0
        for feat in features:
            seg = feat.segment
            overlap = max(0.0, min(ce, seg.end_s) - max(cs, seg.start_s))
            if overlap > 0:
                total += (overlap / clip_len) * feat.saliency
        scores[i] = total
    return starts, scores


def run_tf_selector_stub(
    duration_s: float,
    words: Sequence[WordSpan] | None = None,
    *,
    seed: int = 0,
    cfg: SvHighlightsConfig | None = None,
) -> dict[str, object]:
    c = cfg or SvHighlightsConfig()
    segments = segment_video_stub(duration_s, words, c, seed=seed)
    if words is None:
        rng = np.random.default_rng(seed)
        words = []
        t = 0.0
        while t < duration_s - 1.0:
            cue = rng.choice(list(_HIGHLIGHT_CUES) + ["timeout", "commercial"])
            length = float(rng.uniform(0.2, 0.6))
            words.append(WordSpan(t, t + length, cue))
            t += float(rng.uniform(0.5, 2.5))
    features = build_segment_features(segments, words, seed=seed, cfg=c)
    starts, scores = clip_scores_from_segments(duration_s, features, clip_duration_s=c.clip_duration_s)
    return {
        "n_segments": len(segments),
        "clip_starts_s": starts.tolist(),
        "clip_scores": scores.tolist(),
        "segment_saliency_mean": float(np.mean([f.saliency for f in features])) if features else 0.0,
        "top_segment_score": float(max((f.saliency for f in features), default=0.0)),
    }
