"""X-vector emotion arithmetic for LM-TTS — Brito et al., arXiv:2606.05367."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class XvecEmoConfig:
    paper_arxiv: str = "arXiv:2606.05367"
    title: str = (
        "Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech"
    )
    framework: str = "XVec-Emo-Arithmetic"
    venue: str = "arXiv"
    github: str = "https://github.com/danielbrito91/xvector-emotion-arithmetic"

    backbone: str = "Qwen3-TTS-12Hz-1.7B-Base"
    xvector_dim: int = 2048
    sample_rate_hz: int = 24000

    esd_tau_speakers: tuple[str, ...] = ("0011", "0014", "0017", "0020")
    esd_single_speaker: str = "0017"
    esd_held_out: tuple[str, ...] = ("0013", "0019")
    emouerj_speakers: tuple[str, ...] = ("m03", "m04", "w04")

    utterances_per_speaker_emotion: int = 50
    en_eval_sentences: int = 30

    # Geometry (speaker 0017, angry)
    x_neutral_norm: float = 16.70
    x_angry_norm: float = 16.89
    cos_neutral_angry: float = 0.988
    tau_over_x_ratio: float = 0.154

    # Table 2 avg4spk @ alpha* (EN held-out mean)
    base_eecs_angry: float = 0.539
    avg4spk_eecs_angry: float = 0.925
    avg4spk_eecs_happy: float = 0.687
    avg4spk_eecs_sad: float = 0.761
    avg4spk_secs_mean: float = 0.912
    single_secs_mean: float = 0.810

    delta_eecs_en_avg4spk: float = 0.288
    delta_eecs_en_single: float = 0.291
    delta_eecs_ptbr: float = 0.092

    secs_w_avg4spk_min: float = 0.88
    wer_ptbr_mean: float = 0.006
