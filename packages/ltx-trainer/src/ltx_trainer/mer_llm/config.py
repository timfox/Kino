"""MER-with-LLMs survey config (arXiv:2605.21239)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MerLlmConfig:
    paper_arxiv: str = "arXiv:2605.21239"
    title: str = (
        "Multimodal Emotion Recognition with Large Language Models: "
        "A Survey of the MER-with-LLMs Paradigm"
    )
    corresponding_author: str = "Sicheng Zhao (Tsinghua University)"

    # Five mainstream sub-tasks (Sec. 2, Fig. 3)
    subtasks: tuple[str, ...] = ("GVEC", "VTSA", "SEC", "FER", "CMER")

    # Fig. 1(b) MER-with-LLMs SOTA anchors (representative datasets)
    emoverse_emoset_acc: float = 83.4
    emochat_mvsa_m_acc: float = 72.7
    affectgpt_r1_ov_merd_waf: float = 68.39
    facial_r1_raf_db_acc: float = 92.10
    blsp_emo_meld_acc: float = 57.3
