"""Framework card, benchmark tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.audio_interaction.config import AudioInteractionConfig
from ltx_trainer.audio_interaction.stream import streaming_interaction_loop


def framework_card(cfg: AudioInteractionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioInteractionConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "github": cfg.github,
        "project_page": cfg.project_page,
        "dataset_hub": cfg.dataset_hub,
        "base_model": cfg.base_model,
        "framework": "SoundFlow",
        "loop": "perceive → decide (<silent>|<response>) → respond",
        "chunk_ms": cfg.chunk_ms,
        "streamaudio_2m": {
            "items": cfg.streamaudio_items,
            "hours": cfg.streamaudio_hours,
            "tasks": cfg.streamaudio_tasks,
            "categories": cfg.streamaudio_categories,
        },
        "proactive_sound_bench": cfg.proactive_sound_events,
    }


def streamaudio_categories() -> list[dict[str, Any]]:
    """Figure 6 — StreamAudio-2M task shares."""
    return [
        {"task": "Voice Chatting", "items_k": 539, "share_pct": 23.1},
        {"task": "Streaming Instr. Follow.", "items_k": 487, "share_pct": 20.8},
        {"task": "Streaming Audio Und.", "items_k": 382, "share_pct": 16.4},
        {"task": "Streaming Translation", "items_k": 357, "share_pct": 15.3},
        {"task": "Real-time ASR", "items_k": 270, "share_pct": 11.6},
        {"task": "Proactive Respond", "items_k": 171, "share_pct": 7.3},
        {"task": "Env. Audio Agent", "items_k": 130, "share_pct": 5.5},
    ]


def table1_mmau() -> list[dict[str, Any]]:
    return [
        {"model": "Qwen2.5-Omni-3B", "stream": False, "audio_instruction_avg": 42.51},
        {"model": "Audio-Interaction-3B", "stream": True, "audio_instruction_avg": 58.15},
    ]


def table3_asr_s2tt() -> list[dict[str, Any]]:
    cfg = AudioInteractionConfig()
    return [
        {
            "model": "Audio-Interaction-3B",
            "librispeech_clean_wer": cfg.librispeech_clean_wer,
            "librispeech_other_wer": cfg.librispeech_other_wer,
            "covost_en_zh_bleu": cfg.covost_en_zh_bleu,
            "covost_zh_en_bleu": cfg.covost_zh_en_bleu,
        }
    ]


def table4_proactive_sound() -> list[dict[str, Any]]:
    cfg = AudioInteractionConfig()
    return [
        {"model": "Qwen2.5-Omni-3B", "single_avg": 41.0, "multi_avg": 29.3},
        {"model": "Audio-Interaction-3B", "single_avg": cfg.proactive_single_avg, "multi_avg": cfg.proactive_multi_avg},
    ]


def table5_fifo_inference() -> list[dict[str, Any]]:
    cfg = AudioInteractionConfig()
    return [
        {"setting": "OURS (FIFO)", "fcl_ms": cfg.fifo_fcl_ms, "stall_pct": 0.0},
        {"setting": "w/o FIFO", "fcl_ms": cfg.no_fifo_fcl_ms, "stall_pct": cfg.no_fifo_stall_pct},
    ]


def forward_smoke(cfg: AudioInteractionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioInteractionConfig()
    sr = 16_000
    chunk_samples = int(sr * cfg.chunk_ms / 1000)
    t = chunk_samples * cfg.demo_chunks
    # synthetic stream: quiet → event burst → quiet
    w = torch.zeros(t)
    w[chunk_samples * 2 : chunk_samples * 4] = torch.randn(chunk_samples * 2) * 0.8
    loop = streaming_interaction_loop(w, cfg, sample_rate=sr)
    return {"streaming_loop": loop, "trigger_acc_anchor_pct": cfg.trigger_acc_pct}


def evaluation_demo(cfg: AudioInteractionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioInteractionConfig()
    return {
        "framework": framework_card(cfg),
        "streamaudio_categories": streamaudio_categories(),
        "table1_mmau": table1_mmau(),
        "table3_asr_s2tt": table3_asr_s2tt(),
        "table4_proactive": table4_proactive_sound(),
        "table5_fifo": table5_fifo_inference(),
        "forward": forward_smoke(cfg),
    }
