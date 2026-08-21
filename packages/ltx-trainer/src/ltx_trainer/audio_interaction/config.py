"""Audio-Interaction / SoundFlow — arXiv:2606.05121."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioInteractionConfig:
    paper_arxiv: str = "arXiv:2606.05121"
    title: str = "Audio-Interaction: A Unified Streaming Audio Interaction Model"
    github: str = "https://github.com/xzf-thu/Audio-Interaction"
    project_page: str = "https://xzf-thu.github.io/Audio-Interaction"
    dataset_hub: str = "zhifeixie/StreamAudio-2M"
    base_model: str = "Qwen2.5-Omni-3B"

    chunk_ms: int = 400
    silence_limit_ms: int = 300
    dual_loss_lambda: float = 1.0

    streamaudio_items: int = 2_600_000
    streamaudio_hours: float = 302_000.0
    streamaudio_tasks: int = 28
    streamaudio_categories: int = 7
    proactive_sound_events: int = 644

    # Table 1 — MMAU audio instruction (3B)
    mmau_audio_avg: float = 58.15
    mmau_baseline_omni3b: float = 42.51

    # Table 3 — LibriSpeech WER % / CoVoST2 BLEU
    librispeech_clean_wer: float = 3.17
    librispeech_other_wer: float = 6.04
    covost_en_zh_bleu: float = 55.22
    covost_zh_en_bleu: float = 35.21

    # Table 4 — Proactive-Sound-Bench avg
    proactive_single_avg: float = 61.2
    proactive_multi_avg: float = 62.8

    # Table 5 — FIFO inference
    fifo_fcl_ms: int = 392
    no_fifo_fcl_ms: int = 831
    no_fifo_stall_pct: float = 5.2

    # Ablation V5 trigger accuracy
    trigger_acc_pct: float = 96.77

    demo_chunks: int = 8
