"""IFR, CR, and EMR aggregation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mmae.config import RubricCategory
from ltx_trainer.mmae.sample import MMAESample


def rates_from_rubric_scores(
    sample: MMAESample,
    rubric_scores: list[dict[str, Any]],
) -> dict[str, float]:
    if_scores: list[int] = []
    cr_scores: list[int] = []
    for rubric, scored in zip(sample.rubrics, rubric_scores, strict=True):
        s = int(scored["score"])
        if rubric.category == RubricCategory.INSTRUCTION_FOLLOWING:
            if_scores.append(s)
        else:
            cr_scores.append(s)
    ifr = sum(if_scores) / len(if_scores) if if_scores else 0.0
    cr = sum(cr_scores) / len(cr_scores) if cr_scores else 0.0
    all_scores = [int(s["score"]) for s in rubric_scores]
    emr = 1.0 if all_scores and all(all_scores) else 0.0
    return {"IFR": ifr, "CR": cr, "EMR": emr}


def aggregate_rates(sample_rates: list[dict[str, float]]) -> dict[str, float]:
    if not sample_rates:
        return {"IFR": 0.0, "CR": 0.0, "EMR": 0.0}
    n = len(sample_rates)
    return {
        "IFR": sum(r["IFR"] for r in sample_rates) / n,
        "CR": sum(r["CR"] for r in sample_rates) / n,
        "EMR": sum(r["EMR"] for r in sample_rates) / n,
    }


def aggregate_rates_pct(sample_rates: list[dict[str, float]]) -> dict[str, float]:
    agg = aggregate_rates(sample_rates)
    return {k: round(v * 100.0, 2) for k, v in agg.items()}
