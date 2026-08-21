"""Task taxonomy and domain coverage (VOICEGIRAFFE §3.1)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.voicegiraffe.config import (
    Domain,
    MultiHopTask,
    SingleHopTask,
    TaskTier,
    VoiceGiraffeConfig,
)


def task_taxonomy() -> list[dict[str, Any]]:
    return [
        {
            "tier": TaskTier.SINGLE_HOP.value,
            "task": SingleHopTask.TEMPORAL_LOCALIZATION.value,
            "description": "Precise timestamp retrieval within hour-long streams",
            "avg_duration_min": 52.9,
        },
        {
            "tier": TaskTier.SINGLE_HOP.value,
            "task": SingleHopTask.SEMANTIC_CONTENT.value,
            "description": "Factual comprehension and topic tracking",
            "avg_duration_min": 52.4,
        },
        {
            "tier": TaskTier.SINGLE_HOP.value,
            "task": SingleHopTask.ACOUSTIC_EVENT.value,
            "description": "Non-speech sound and music recognition",
            "avg_duration_min": 53.7,
        },
        {
            "tier": TaskTier.SINGLE_HOP.value,
            "task": SingleHopTask.PARALINGUISTIC.value,
            "description": "Speaker emotion, age, gender, timbre, pitch beyond transcript",
            "avg_duration_min": 53.9,
        },
        {
            "tier": TaskTier.MULTI_HOP.value,
            "task": MultiHopTask.CAUSAL_ALIGNMENT.value,
            "description": "Causal chains from distributed evidence across segments",
            "avg_duration_min": 55.6,
        },
        {
            "tier": TaskTier.MULTI_HOP.value,
            "task": MultiHopTask.EVENT_TRACKING.value,
            "description": "Sparse event aggregation and long-term memory stress test",
            "avg_duration_min": 55.1,
        },
    ]


def domain_coverage() -> list[dict[str, str]]:
    return [
        {"domain": Domain.SPORTS.value, "modalities": "speech, dense temporal events"},
        {"domain": Domain.ESPORTS.value, "modalities": "speech, team comms, chaotic multi-speaker"},
        {"domain": Domain.TV_DRAMA.value, "modalities": "speech, SFX, background music"},
        {"domain": Domain.NEWS.value, "modalities": "speech, formal register, topic transitions"},
        {"domain": Domain.PODCAST.value, "modalities": "speech, conversational turn-taking"},
    ]


def benchmark_stats(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    return {
        "total_recordings": cfg.n_recordings,
        "total_audio_hours": cfg.total_hours,
        "avg_duration_min": cfg.avg_duration_min,
        "pct_over_one_hour": cfg.pct_over_one_hour,
        "sub_tasks": cfg.n_sub_tasks,
        "data_domains": cfg.n_domains,
        "languages": list(cfg.languages),
        "total_qa_items": cfg.n_qa_total,
        "single_hop_items": cfg.n_single_hop,
        "multi_hop_items": cfg.n_multi_hop,
    }
