"""PDE survey smoke (arXiv:2605.26133)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pde_survey.attacks import (
    contamination_flag,
    losses_from_texts,
    membership_infer,
    min_k_percentile_gap,
    ngram_overlap_fraction,
)
from ltx_trainer.pde_survey.config import PdeSurveyConfig
from ltx_trainer.pde_survey.defenses import defense_catalog
from ltx_trainer.pde_survey.exposure import (
    dataset_fully_exposed,
    dataset_partially_exposed,
    exposure_score,
    instance_exposed,
)
from ltx_trainer.pde_survey.tables import headline_results, table2_sota_availability


def evaluation_smoke(cfg: PdeSurveyConfig | None = None) -> dict[str, Any]:
    c = cfg or PdeSurveyConfig()
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "benchmark contamination inflates llm scores",
    ]
    eval_set = [
        "the quick brown fox jumps over the lazy dog",
        "unseen evaluation prompt about quantum optics",
    ]

    inst_flags = [instance_exposed("M", x, corpus) for x in eval_set]
    pde_score = exposure_score("M", eval_set, corpus)
    partial = dataset_partially_exposed("M", eval_set, corpus)
    full = dataset_fully_exposed("M", eval_set, corpus)

    member_text = "the quick brown fox " * 8
    nonmember_text = "zyxwvutsrqponmlkjihgfedcba rare token salad"
    mia_m = membership_infer(member_text, member_ceiling=c.member_ppl_ceiling)
    mia_n = membership_infer(nonmember_text, member_ceiling=c.member_ppl_ceiling)

    overlap = ngram_overlap_fraction(eval_set[0], corpus[0], n=4)
    contaminated = contamination_flag(overlap)

    mem_losses = losses_from_texts([member_text], member_like=True)
    non_losses = losses_from_texts([nonmember_text], member_like=False)
    min_k_gap = min_k_percentile_gap(mem_losses, non_losses)

    t2 = table2_sota_availability()
    code_available_count = sum(1 for row in t2 if row["code"])

    return {
        "paper": c.paper_arxiv,
        "instance_flags": inst_flags,
        "exposure_score": round(pde_score, 3),
        "dataset_partially_exposed": partial,
        "dataset_fully_exposed": full,
        "mia_member_predicted": mia_m.predicted_member,
        "mia_nonmember_predicted": mia_n.predicted_member,
        "ngram_overlap": round(overlap, 3),
        "contamination_flag": contaminated,
        "min_k_gap": round(min_k_gap, 4),
        "defense_categories": len(defense_catalog()),
        "table2_rows": len(t2),
        "table2_code_available_rows": code_available_count,
        "paper_unified_contribution": headline_results()["unified_framework"],
        "scenarios_count": len(c.scenarios),
    }
