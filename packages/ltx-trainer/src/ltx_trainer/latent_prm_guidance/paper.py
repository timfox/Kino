"""Paper cards and knowledge bundle."""

from __future__ import annotations

from typing import Any

from ltx_trainer.latent_prm_guidance.benchmarks import (
    paired_significance,
    summary_anchors,
    table_1_main,
    table_2_branch_selection,
    table_5_directions,
    table_6_ablation,
    trajectory_analysis,
)
from ltx_trainer.latent_prm_guidance.constants import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, UPSTREAM_REPO
from ltx_trainer.latent_prm_guidance.latent import alignment_transform_card
from ltx_trainer.latent_prm_guidance.paratrans import paratrans_card
from ltx_trainer.latent_prm_guidance.prm import inference_card, prm_architecture_card, training_card
from ltx_trainer.latent_prm_guidance.references import reference_anchors
from ltx_trainer.latent_prm_guidance.reward import reward_card


def paper_card() -> dict[str, Any]:
    return {
        "arxiv": PAPER_ARXIV,
        "title": PAPER_TITLE,
        "url": PAPER_URL,
        "upstream": UPSTREAM_REPO,
        "authors": "Bitan, Kaplan, Bar-Yadin, Ghrayeb, Chen, Jhaveri, Hasabnis, Oren",
        "findings": [
            "Pre-decoding latent PRM branch selection improves ParaTrans validation +9.21 pp",
            "Frozen LLaMA-3.3-70B + smaller Qwen-Coder-7B PRM over hidden prefixes",
            "Gains persist under 3-attempt repair; beats post-decode text-PRM reranking",
            "Branch-selection test supports filtering harmful perturbations locally",
        ],
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "paper": paper_card(),
        "paratrans": paratrans_card(),
        "latent_alignment": alignment_transform_card(),
        "prm": {
            "architecture": prm_architecture_card(),
            "training": training_card(),
            "inference": inference_card(),
        },
        "reward": reward_card(),
        "table_1": table_1_main(),
        "table_2": table_2_branch_selection(),
        "table_5_directions": table_5_directions(),
        "table_6_ablation": table_6_ablation(),
        "paired_significance": paired_significance(),
        "trajectory_analysis": trajectory_analysis(),
        "summary": summary_anchors(),
        "references": reference_anchors(),
    }
