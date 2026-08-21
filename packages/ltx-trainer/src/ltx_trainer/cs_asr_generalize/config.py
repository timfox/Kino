"""CS-ASR generalization to unseen language pairs — Paik et al., arXiv:2606.05846."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CsAsrGeneralizeConfig:
    paper_arxiv: str = "arXiv:2606.05846"
    title: str = (
        "Towards Truly Multilingual ASR: Generalizing Code-Switching ASR "
        "to Unseen Language Pairs"
    )
    framework: str = "CS-ASR-Generalize"

    backbone: str = "WHISPER-MEDIUM"
    metric: str = "MER"
    languages: tuple[str, ...] = ("EN", "KO", "JA", "DE")
    seen_pairs: tuple[str, ...] = ("KO-EN", "JA-EN", "DE-EN")
    unseen_pairs: tuple[str, ...] = ("KO-DE", "KO-JA")

    # Evaluation sets (Sec 3.1)
    ko_ja_eval_utterances: int = 450
    ko_de_eval_utterances: int = 387
    ko_ja_hub: str = "thetaone-ai/Korean-Japanese-Code-Switching-Speech"

    # Table 1 — WHISPER-MEDIUM baseline
    baseline_ko_en: float = 0.26
    baseline_ja_en: float = 0.56
    baseline_de_en: float = 0.15
    baseline_seen_avg: float = 0.33
    baseline_ko_de: float = 0.39
    baseline_ko_ja: float = 0.44
    baseline_unseen_avg: float = 0.41

    # Table 1 — best merging (TIES, all three seen pairs)
    ties3_seen_avg: float = 0.14
    ties3_unseen_avg: float = 0.34
    ties3_ko_de: float = 0.37
    ties3_ko_ja: float = 0.30

    # Table 1 — best DG (Fishr)
    fishr_seen_avg: float = 0.18
    fishr_unseen_avg: float = 0.33
    fishr_ko_de: float = 0.35
    fishr_ko_ja: float = 0.31

    # Table 1 — pairwise TIES KO-EN + JA-EN (strong seen/unseen tradeoff)
    ties_ko_ja_en_seen_avg: float = 0.14
    ties_ko_ja_en_unseen_avg: float = 0.32

    # Limitations anchor (Sec 5)
    limitation_unseen_mer_floor: float = 0.32

    @property
    def n_languages(self) -> int:
        return len(self.languages)

    @property
    def possible_pairs(self) -> int:
        n = self.n_languages
        return n * (n - 1) // 2
