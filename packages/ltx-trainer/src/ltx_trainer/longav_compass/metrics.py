"""Video, audio, and task-specific metrics (Sec. 3.5–3.7)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class VideoMetric(str, Enum):
    VQA = "VQA"  # event fulfillment
    VQ = "VQ"  # visual quality 1-5
    CONT = "Cont."  # long-form continuity 1-5
    TRANS = "Trans."  # transition stability 1-5
    HOL = "Hol."  # holistic presentation 1-5
    TVALIGN = "TVAlign"  # CLIP text-video 0-1


class AudioMetric(str, Enum):
    AVS = "AVS"  # audio-video sync 1-5
    AUDQ = "AudQ"  # audio quality 1-5
    AUDL = "AudL"  # long-audio coherence 1-5


class I2AVMetric(str, Enum):
    IV1 = "IV1"  # first-frame anchoring 0-1
    IMGALIGN = "ImgAlign"  # CLIP image-video 0-1


MOS_ANCHORS: dict[int, str] = {
    1: "failed / severe defects",
    2: "poor with major issues",
    3: "acceptable with visible issues",
    4: "good with minor issues",
    5: "excellent",
}


@dataclass
class VQSubscores:
    motion_naturalness: float
    subject_integrity: float
    artifact_control: float
    visual_fidelity: float

    def mean(self) -> float:
        return (self.motion_naturalness + self.subject_integrity + self.artifact_control + self.visual_fidelity) / 4.0


def score_event_vqa(qa_answers: list[str]) -> float:
    """Event fulfillment: yes=1.0, partial=0.5, no=0.0."""
    mapping = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    if not qa_answers:
        return 0.0
    return sum(mapping.get(a.lower(), 0.0) for a in qa_answers) / len(qa_answers)


def duration_weighted_average(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if w.sum() <= 0:
        return float(v.mean())
    return float((v * w).sum() / w.sum())


def clip_text_video_alignment(
    text: str,
    frame_features: list[np.ndarray],
    *,
    text_feature: np.ndarray | None = None,
) -> float:
    """TVAlign proxy: mean cosine similarity (unit-norm features)."""
    if not frame_features:
        return 0.0
    if text_feature is None:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        text_feature = rng.standard_normal(512)
    t = text_feature / (np.linalg.norm(text_feature) + 1e-8)
    sims = []
    for f in frame_features:
        v = f / (np.linalg.norm(f) + 1e-8)
        sims.append(float(np.clip(np.dot(t, v), -1.0, 1.0)))
    return float(np.mean([(s + 1.0) / 2.0 for s in sims]))  # map to 0-1


def clip_image_alignment(
    reference_feature: np.ndarray,
    frame_features: list[np.ndarray],
) -> float:
    """ImgAlign proxy: trimmed mean cosine similarity."""
    if not frame_features:
        return 0.0
    ref = reference_feature / (np.linalg.norm(reference_feature) + 1e-8)
    sims = sorted(float(np.dot(ref, f / (np.linalg.norm(f) + 1e-8))) for f in frame_features)
    trim = max(0, len(sims) // 10)
    core = sims[trim : len(sims) - trim] if len(sims) > 2 * trim else sims
    return float(np.mean([(s + 1.0) / 2.0 for s in core]))


def transition_algorithm_score(
    *,
    black_frames: bool = False,
    flicker: bool = False,
    freeze: bool = False,
    repetition: bool = False,
) -> float:
    """Algorithmic transition checks combined with 1-5 scale."""
    defects = sum([black_frames, flicker, freeze, repetition])
    return float(max(1.0, 5.0 - defects * 1.25))


def balanced_score(
    metric_values: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Composite 0–100 balanced score (diagnostic leaderboard helper)."""
    if not metric_values:
        return 0.0
    weights = weights or {}
    total_w = 0.0
    total = 0.0
    for name, val in metric_values.items():
        w = weights.get(name, 1.0)
        # Normalize MOS 1-5 → 0-100; 0-1 metrics → 0-100
        if name in {m.value for m in VideoMetric if m != VideoMetric.TVALIGN} | {m.value for m in AudioMetric}:
            scaled = (val - 1.0) / 4.0 * 100.0 if val <= 5.0 else val
        else:
            scaled = val * 100.0 if val <= 1.0 else val
        total += scaled * w
        total_w += w
    return round(total / total_w if total_w else 0.0, 1)


def list_video_metric_definitions() -> list[dict[str, Any]]:
    return [
        {"id": VideoMetric.VQA.value, "scale": "0-1", "level": "event"},
        {"id": VideoMetric.VQ.value, "scale": "1-5", "level": "event"},
        {"id": VideoMetric.CONT.value, "scale": "1-5", "level": "full-video"},
        {"id": VideoMetric.TRANS.value, "scale": "1-5", "level": "boundary"},
        {"id": VideoMetric.HOL.value, "scale": "1-5", "level": "full-video"},
        {"id": VideoMetric.TVALIGN.value, "scale": "0-1", "level": "full-video"},
    ]


def list_audio_metric_definitions() -> list[dict[str, Any]]:
    return [
        {"id": AudioMetric.AVS.value, "scale": "1-5"},
        {"id": AudioMetric.AUDQ.value, "scale": "1-5"},
        {"id": AudioMetric.AUDL.value, "scale": "1-5"},
    ]
