"""Framework card and benchmarks bundle for MER-with-LLMs survey."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mer_llm.challenges import three_challenges
from ltx_trainer.mer_llm.config import MerLlmConfig
from ltx_trainer.mer_llm.formulation import mer_with_llms_response_schema
from ltx_trainer.mer_llm.layout import LIMITATIONS
from ltx_trainer.mer_llm.mock import evaluation_smoke
from ltx_trainer.mer_llm.scoring import explainable_mer_stub
from ltx_trainer.mer_llm.tables import (
    fig1b_paradigm_progress,
    table1_emotion_datasets_excerpt,
    table2_perceptual_mapping_excerpt,
    table5_quantitative_excerpt,
)
from ltx_trainer.mer_llm.catalog import research_radar_brief
from ltx_trainer.mer_llm.taxonomy import five_subtasks, future_directions, taxonomy_branches


def framework_card(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MerLlmConfig()
    return {
        "name": cfg.title,
        "paper": cfg.paper_arxiv,
        "authors": (
            "Hongrui Zhang, Daiqing Wu, Yangyang Li, Kuien Liu, Yuhui Wang, "
            "Yu Zhou, Sicheng Zhao"
        ),
        "corresponding": cfg.corresponding_author,
        "paradigm": "MER-with-LLMs — MLLM-centric autoregressive emotion understanding",
        "contributions": [
            "Three challenge branches: data scarcity, affective gap, interpretation opacity.",
            "Taxonomy: Affective Data Augmentation, Representation, Reasoning (Fig. 2).",
            "Five sub-tasks: GVEC, VTSA, SEC, FER, CMER with modality formulations.",
            "Tables 1–5 and Fig. 1 progress anchors; future directions for unified MER.",
        ],
        "challenges": three_challenges(),
        "taxonomy": taxonomy_branches(),
        "subtasks": five_subtasks(),
        "response_schema": mer_with_llms_response_schema(),
        "future_directions": future_directions(),
        "headlines": {
            "emoverse_emoset_acc_pct": cfg.emoverse_emoset_acc,
            "affectgpt_r1_ov_merd_waf_pct": cfg.affectgpt_r1_ov_merd_waf,
            "facial_r1_raf_db_acc_pct": cfg.facial_r1_raf_db_acc,
        },
        "limitations": LIMITATIONS,
    }


def evaluation_demo(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MerLlmConfig()
    gvec = next(r for r in fig1b_paradigm_progress() if r["sub_task"] == "GVEC")
    return {
        "paper": cfg.paper_arxiv,
        "smoke": evaluation_smoke(cfg),
        "explainable_example": explainable_mer_stub(
            observation="Two actors demonstrate skincare steps; tone is promotional yet warm.",
            emotion="satisfied, contentment, happy",
            confidence=0.88,
        ),
        "fig1b_gvec_row": gvec,
    }


def benchmarks_bundle(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MerLlmConfig()
    from ltx_trainer.mer_llm.catalog import methods_for_subtask, representative_methods

    return {
        "framework": framework_card(cfg),
        "fig1b_paradigm_progress": fig1b_paradigm_progress(),
        "table1_datasets_excerpt": table1_emotion_datasets_excerpt(),
        "table2_perceptual_mapping_excerpt": table2_perceptual_mapping_excerpt(),
        "table5_quantitative_excerpt": table5_quantitative_excerpt(),
        "representative_methods": representative_methods(),
        "cmer_methods": methods_for_subtask("CMER"),
        "research_radar_game_live_ops": research_radar_brief(vertical="game_live_ops"),
        "limitations": LIMITATIONS,
    }


def training_step_demo(cfg: MerLlmConfig | None = None) -> dict[str, Any]:
    """Suggested fine-tuning narrative (Emotion-LLaMA / AffectGPT style)."""
    cfg = cfg or MerLlmConfig()
    return {
        "paper": cfg.paper_arxiv,
        "recipe": [
            "Curate emotion-oriented instruction data (Table 1 engineering branch).",
            "Align encoders via perceptual mapping or coordination (Tables 2–3).",
            "Optional SFT+RL for explanation consistency or open-vocabulary (Table 4).",
            "Evaluate on sub-task benchmarks (Table 5); report WAF/UAR/Acc as appropriate.",
        ],
        "recommended_anchors": {
            "GVEC": "EmoVIT on EmoSet",
            "CMER": "AffectGPT-R1 on OV-MERD+",
            "FER": "Facial-R1 on RAF-DB",
        },
    }
