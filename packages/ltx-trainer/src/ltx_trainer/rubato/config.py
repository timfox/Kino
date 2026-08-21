"""Configuration for Rubato / InterMo (arXiv:2605.24291)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RubatoConfig:
    paper_arxiv: str = "arXiv:2605.24291"
    demo_url: str = "https://nctamer.github.io/rubato-transcription"

    # Architecture (Sec. 3, Fig. 2).
    backbone_family: str = "Canary-180M-Flash-style"
    approx_params_m: int = 180
    encoder_frame_hz: float = 25.0
    audio_sample_rate_hz: int = 16_000
    inference_window_seconds: float = 40.0
    inference_hop_fraction: float = 0.5

    # Timestamps (Sec. 3.3).
    timestamp_bin_ms: int = 10
    timestamp_bins_per_window: int = 4_000  # 0.00–40.00 s at 10 ms
    label_smoothing_p_center: float = 0.9
    label_smoothing_window_bins: int = 5  # w in paper; ±5 bins → 110 ms span

    # Subword regularization (Sec. 3.3).
    subword_regularization_alpha: float = 0.25

    # Table 1 — raw audio hours (approximate from paper).
    maestro_hours: float = 159.0
    nasap_hours: float = 30.0
    pdmx_synth_hours: float = 2071.0

    # RTFx (Sec. 3.3 inference note).
    rtfx_amt: float = 9.0
    rtfx_tast: float = 21.0
    rtfx_dbd: float = 112.0

    # Vocabulary (Sec. 3.2).
    vocab_total: int = 8_000
    vocab_semantic_approx: int = 3570

    dialects_inference: tuple[str, ...] = ("TAST", "AMT", "DBD")
    dialects_training_only: tuple[str, ...] = ("TASTlite", "A2S", "A2Slite", "AMTlite", "DBDplus")

    export_formats: tuple[str, ...] = ("Humdrum", "MEI", "MusicXML")
    renderer: str = "Verovio"


@dataclass
class InterMoDialect:
    """Prompt-conditioned dialect (Fig. 2 bottom)."""

    name: str
    prompt_tokens: tuple[str, ...]
    description: str
    inference: bool = False


def dialect_registry() -> list[InterMoDialect]:
    return [
        InterMoDialect(
            "TAST",
            ("<|piano|>", "<|score|>", "<|timestamp|>", "<|spell|>"),
            "Time-aligned score transcription: full notation + timestamps + pitch spelling",
            inference=True,
        ),
        InterMoDialect(
            "TASTlite",
            ("<|piano|>", "<|score|>", "<|timestamp|>"),
            "TAST without <|spell|> (MIDI-like pitch names)",
        ),
        InterMoDialect(
            "A2S",
            ("<|piano|>", "<|score|>", "<|spell|>"),
            "Audio-to-score without timestamps (PDMX-scale)",
        ),
        InterMoDialect(
            "A2Slite",
            ("<|piano|>", "<|score|>"),
            "A2S without pitch spelling",
        ),
        InterMoDialect(
            "AMTlite",
            ("<|piano|>", "<|timestamp|>"),
            "Onsets with timestamps, no MIDI controls",
        ),
        InterMoDialect(
            "AMT",
            ("<|piano|>", "<|timestamp|>", "<|MIDI|>"),
            "MIDI-like: velocity <|vel:N|>, sustain <|CC64:on/off|>",
            inference=True,
        ),
        InterMoDialect(
            "DBDplus",
            ("<|beats|>", "<|score|>", "<|timestamp|>"),
            "Downbeats/beats with meter and key on barlines",
        ),
        InterMoDialect(
            "DBD",
            ("<|beats|>", "<|timestamp|>"),
            "Beat * and downbeat | markers only",
            inference=True,
        ),
    ]


@dataclass
class RubatoTrainingMix:
    """Table 1 style coverage (flags only; k-counts omitted in stub)."""

    source: str
    audio_hours: float
    tast: bool
    a2s: bool
    amt: bool
    dbd: bool


def training_data_rows() -> list[RubatoTrainingMix]:
    return [
        RubatoTrainingMix("MAESTRO", 159.0, False, False, True, False),
        RubatoTrainingMix("(n)ASAP", 30.0, True, True, False, True),
        RubatoTrainingMix("PDMX", 2071.0, True, True, True, True),
    ]
