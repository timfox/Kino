"""Dataset construction: alignment (~700K) + SFT (~100K) per Sec. 4.1 and Supp. 7.6."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ltx_trainer.autocut.config import AutoCutConfig
from ltx_trainer.autocut.sft import (
    build_bgm_selection_example,
    build_script_generation_example,
    build_video_selection_example,
    build_video_sorting_example,
)
from ltx_trainer.autocut.taxonomy import EditingTask
from ltx_trainer.autocut.tokens import serialize_alignment_sample


@dataclass
class AdRecord:
    """Parsed advertisement record after ASR + segmentation."""

    video_id: str
    product_type: str
    brand: str
    features: list[str]
    script_lines: list[str]
    clip_durations_s: list[float]
    video_duration_s: float
    ctr_percentile: float = 0.95
    like_rate_percentile: float = 0.95
    has_speech: bool = True
    lyrical_only: bool = False
    clip_relevance_scores: list[int] = field(default_factory=list)


@dataclass
class ParsedClip:
    index: int
    asr_text: str
    start_s: float
    end_s: float
    video_tokens: list[str]
    is_positive: bool = True


def parse_asr_into_clips(transcript: str) -> list[tuple[str, float, float]]:
    """Split ASR by punctuation into clip spans (Fig. 7)."""
    sentences = re.split(r"(?<=[.!?。！？])\s*", transcript.strip())
    clips: list[tuple[str, float, float]] = []
    t = 0.0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        dur = max(1.0, min(60.0, len(s.split()) * 0.35))
        clips.append((s, t, t + dur))
        t += dur
    return clips


def filter_alignment_record(rec: AdRecord, cfg: AutoCutConfig | None = None) -> bool:
    """Alignment filters: engagement, speech, clip/video duration (Supp. 7.6.1)."""
    c = cfg or AutoCutConfig()
    if rec.ctr_percentile < 0.9 or rec.like_rate_percentile < 0.9:
        return False
    if rec.lyrical_only or not rec.has_speech:
        return False
    n_clips = len(rec.script_lines)
    if n_clips < 1 or n_clips >= 60:
        return False
    if rec.video_duration_s < 1 or rec.video_duration_s > 60:
        return False
    for d in rec.clip_durations_s:
        if d < 1 or d > 60:
            return False
    return True


def filter_sft_record(rec: AdRecord, *, min_relevance: int = 4, min_fraction: float = 0.8) -> bool:
    """SFT filters: duration, caption quality, visual-text relevance (Supp. 7.6.2)."""
    if rec.video_duration_s > 120:
        return False
    if not rec.script_lines or any(len(s.strip()) < 2 for s in rec.script_lines):
        return False
    for d in rec.clip_durations_s:
        if d < 2 or d > 60:
            return False
    if rec.clip_relevance_scores:
        good = sum(1 for s in rec.clip_relevance_scores if s >= min_relevance)
        if good / len(rec.clip_relevance_scores) < min_fraction:
            return False
    return True


def dedupe_hash(rec: AdRecord) -> str:
    payload = f"{rec.brand}|{rec.product_type}|" + "|".join(rec.script_lines)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_alignment_sample(
    rec: AdRecord,
    *,
    video_token_rows: list[list[str]] | None = None,
    audio_tokens: list[str] | None = None,
) -> str:
    rows = video_token_rows or [[f"<|video_{i % 8}_{j % 16}|>"] for i, _ in enumerate(rec.script_lines) for j in range(1)]
    audio = audio_tokens or [f"<|audio_{h}_{h % 16}|>" for h in range(8)]
    return serialize_alignment_sample(
        script_blocks=rec.script_lines,
        video_token_rows=rows[: len(rec.script_lines)],
        audio_tokens=audio,
    )


def build_sft_instances_for_record(
    rec: AdRecord,
    *,
    candidate_pool: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Four task instances per filtered record (Supp. 7.6.3)."""
    product = (
        f"Product Type: {rec.product_type}\nBrand: {rec.brand}\n"
        f"Features: [{', '.join(rec.features)}]"
    )
    script = "\n".join(rec.script_lines)
    n = len(rec.script_lines)
    pool = list(candidate_pool or range(max(n * 2, 4)))
    selected = list(range(min(n, len(pool))))
    shuffled = selected.copy()
    shuffled.reverse()

    return {
        "video_selection": build_video_selection_example(
            product_info=product,
            script=script,
            candidate_indices=pool,
            selected_indices=selected,
        ).to_sharegpt(),
        "video_sorting": build_video_sorting_example(
            product_info=product,
            script=script,
            shuffled_indices=shuffled,
            sorted_indices=selected,
        ).to_sharegpt(),
        "script_generation": build_script_generation_example(
            product_info=product,
            clip_count=n,
            script_lines=rec.script_lines,
        ).to_sharegpt(),
        "bgm_selection": build_bgm_selection_example(
            product_info=product,
            script=script,
            audio_token_line=" ".join(f"<|audio_{h}_{h % 12}|>" for h in range(8)),
        ).to_sharegpt(),
    }


def dataset_pipeline_summary(cfg: AutoCutConfig | None = None) -> dict[str, Any]:
    c = cfg or AutoCutConfig()
    return {
        "alignment": {
            "target_samples": c.alignment_samples,
            "filters": [
                "top 10% CTR and like-rate",
                "no lyrical-only / no-speech",
                "1 < clips < 60, 1s < clip < 60s, 1s < video < 60s",
            ],
            "parsing": "1 fps frames, PANNs BGM, ASR punctuation clips",
        },
        "sft": {
            "target_samples": c.sft_samples,
            "filters": [
                "video < 120s",
                "clip 2–60s",
                "≥80% clips Qwen2.5-VL relevance ≥ 4/5",
                "dedupe brand+product+script",
            ],
            "tasks": [t.value for t in EditingTask],
            "instances_per_task": "25K–30K",
        },
    }
