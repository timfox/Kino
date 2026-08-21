"""LiveK12Bench configuration (arXiv:2605.26781)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Livek12Config:
    paper_arxiv: str = "2605.26781"
    paper_title: str = (
        "LiveK12Bench: Have Large Multimodal Models Truly Conquered "
        "High School-level Examinations?"
    )
    authors: str = (
        "Xiaohan Wang, Mingze Yin, Yilin Zhao, Gang Liu, Dian Li (Tencent PCG / ZJU)"
    )

    n_questions: int = 2_114
    n_knowledge_points: int = 2_725
    disciplines: tuple[str, ...] = ("mathematics", "physics", "chemistry", "biology")
    modalities: tuple[str, ...] = ("text_only", "text_image", "image_only_exam")

    # Mock-exam hyperparameters (paper §3.2)
    process_penalty_tau: float = 3.0
    efficiency_lambda: float = 0.15
    process_weight_wp: float = 0.5
    avg_response_length_L_bar: int = 4096
    # Figure 1 headline drops (default → exam-realistic, ×100 scale)
    gpt5_default_oes: float = 79.0
    gpt5_exam_oes: float = 53.0
