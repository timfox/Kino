"""SpeechJBB — LALM code-switched audio jailbreak benchmark (arXiv:2606.06037)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeechJbbConfig:
    paper_arxiv: str = "arXiv:2606.06037"
    title: str = (
        "SpeechJBB: Probing Safety Alignment and Comprehension in "
        "Large Audio Language Models under Code-Switched Speech"
    )
    framework: str = "SPEECHJBB"
    affiliations: tuple[str, ...] = ("Mila", "McGill University")

    # Dataset (§3)
    harmful_prompts: int = 100
    benign_prompts: int = 100
    languages: tuple[str, ...] = ("en", "de", "es", "fr", "it")
    code_switch_pairs: tuple[str, ...] = (
        "en-de",
        "en-es",
        "en-fr",
        "en-it",
        "de-es",
        "de-fr",
        "fr-it",
        "es-it",
        "es-fr",
        "de-it",
    )
    pseudo_word_ratios: tuple[float, ...] = (0.10, 0.30, 0.50)
    n_models: int = 9

    # Table 3 — mean across nine LALMs (%)
    mono_rr: float = 81.54
    mono_dr: float = 2.00
    mono_jsr: float = 16.39
    enx_rr: float = 79.32
    enx_dr: float = 3.67
    enx_jsr: float = 17.01
    xy_rr: float = 69.76
    xy_dr: float = 9.28
    xy_jsr: float = 20.92
    mean_jsr: float = 18.37

    # Table 3 — model extremes
    gemini_mean_jsr: float = 4.76
    voxtral_mean_jsr: float = 48.27
    proprietary_mean_jsr: float = 7.9
    opensource_mean_jsr: float = 21.3

    # Table 4 — pseudo-word obfuscation mean JSR (%)
    pseudo_10_jsr: float = 20.3
    pseudo_30_jsr: float = 22.5
    pseudo_50_jsr: float = 24.6
    pseudo_50_xy_jsr: float = 25.48

    # Table 1 — monolingual synthesis quality
    en_wer: float = 5.4
    en_utmos: float = 4.2

    # Table 6 — MGSM accuracy (%)
    gemini_mgsm: float = 97.9
    voxtral_mgsm: float = 72.9
