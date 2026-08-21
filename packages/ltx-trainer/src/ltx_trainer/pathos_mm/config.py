"""Multimodal pathos analysis stub (arXiv:2605.22732)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PathosMmConfig:
    paper_arxiv: str = "arXiv:2605.22732"
    title: str = "Beyond acoustic emotion recognition: multimodal pathos analysis in political speech"
    banaszak_segments_total: int = 51
    banaszak_segments_analyzed: int = 41
    emo_db_utterances: int = 535
    # Headline Spearman correlations (Table 4)
    rho_gemini_valence_trust: float = 0.664
    rho_gemini_arousal_trust: float = -0.535
    rho_e2v_valence_trust: float = 0.097
    rho_e2v_arousal_trust: float = -0.155
    # EMO-DB Gemini open-ended match (Table 2)
    emo_db_overall_match_pct: float = 30.1
