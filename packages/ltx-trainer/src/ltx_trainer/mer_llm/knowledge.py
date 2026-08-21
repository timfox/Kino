"""Agent-facing MER-with-LLMs survey facts (no weights download)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mer_llm.config import MerLlmConfig
from ltx_trainer.mer_llm.taxonomy import five_subtasks, future_directions, taxonomy_branches


def mer_llm_knowledge_blob(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MerLlmConfig()
    from ltx_trainer.mer_llm.challenges import three_challenges
    from ltx_trainer.mer_llm.tables import fig1b_paradigm_progress

    return {
        "name": "MER-with-LLMs",
        "paper": cfg.title,
        "arxiv": cfg.paper_arxiv,
        "corresponding": cfg.corresponding_author,
        "paradigm": "MLLM-centric autoregressive multimodal emotion understanding",
        "subtasks": [s["id"] for s in five_subtasks()],
        "challenges": [c["name"] for c in three_challenges()],
        "taxonomy_branches": [b["name"] for b in taxonomy_branches()],
        "future_directions": list(future_directions()),
        "headline_results": {
            "EmoVIT_EmoSet_Acc_pct": cfg.emoverse_emoset_acc,
            "EmoChat_MVSA-M_Acc_pct": cfg.emochat_mvsa_m_acc,
            "Facial-R1_RAF-DB_Acc_pct": cfg.facial_r1_raf_db_acc,
            "AffectGPT-R1_OV-MERD+_WAF_pct": cfg.affectgpt_r1_ov_merd_waf,
            "BLSP-Emo_MELD_Acc_pct": cfg.blsp_emo_meld_acc,
        },
        "fig1b_progress": fig1b_paradigm_progress(),
        "related_gopex": {
            "pathos_mm": "Political speech pathos vs acoustic SER (arXiv:2605.22732)",
            "longav_compass": "Minute-scale AV generation eval, not MER labels (arXiv:2605.26244)",
            "avbench": "Short T2AV human-aligned automated metrics",
            "audio_reasoning_survey": "Audio reasoning in MLLMs (arXiv:2605.21008)",
        },
        "gopex_snippet": (
            "from ltx_trainer.mer_llm import framework_card, evaluation_demo\n"
            "print(framework_card()['challenges'])\n"
            "print(evaluation_demo()['explainable_example'])\n"
        ),
    }
