"""Mock-exam scoring formulas (paper §2.3)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.livek12.config import Livek12Config


def process_score(
    point_value: float,
    errors: dict[str, int],
    tau: float | None = None,
) -> float:
    """P_i = V_i - tau * sum_k x_{i,k}."""
    t = tau if tau is not None else Livek12Config().process_penalty_tau
    penalty = sum(int(errors.get(k, 0)) for k in ("CIE", "LAE", "DRE"))
    return max(0.0, point_value - t * penalty)


def exam_score_item(
    point_value: float,
    outcome_points: float,
    process_points: float,
    wp: float | None = None,
) -> float:
    """ES_i = wp * P_i * (O_i/V_i) + (1-wp) * O_i."""
    w = wp if wp is not None else Livek12Config().process_weight_wp
    if point_value <= 0:
        return 0.0
    prop = outcome_points / point_value
    return w * process_points * prop + (1.0 - w) * outcome_points


def overall_exam_score(exam_scores: list[float], point_values: list[float]) -> float:
    """OES on 100-point scale."""
    total_v = sum(point_values)
    if total_v <= 0:
        return 0.0
    return 100.0 * sum(exam_scores) / total_v


def arl(
    accuracies: list[int],
    lengths: list[int],
    l_bar: int | None = None,
    lam: float | None = None,
) -> float:
    """ARL = mean_i S_i * (1 + lambda * ln(L_bar / l_i))."""
    if not accuracies:
        return 0.0
    lb = l_bar if l_bar is not None else Livek12Config().avg_response_length_L_bar
    la = lam if lam is not None else Livek12Config().efficiency_lambda
    terms = []
    for s, li in zip(accuracies, lengths, strict=True):
        if li <= 0:
            terms.append(float(s))
            continue
        terms.append(s * (1.0 + la * math.log(lb / li)))
    return sum(terms) / len(terms)


def mock_exam_scoring_smoke() -> dict[str, Any]:
    """Toy two-question exam illustrating OES vs accuracy-only."""
    cfg = Livek12Config()
    items = [
        {"V": 10.0, "O": 10.0, "errors": {"CIE": 0, "LAE": 0, "DRE": 0}},
        {"V": 15.0, "O": 5.0, "errors": {"CIE": 1, "LAE": 0, "DRE": 1}},
    ]
    rows = []
    es_sum = 0.0
    v_sum = 0.0
    acc = 0
    for it in items:
        p = process_score(it["V"], it["errors"], cfg.process_penalty_tau)
        es = exam_score_item(it["V"], it["O"], p, cfg.process_weight_wp)
        es_sum += es
        v_sum += it["V"]
        acc += 1 if it["O"] >= it["V"] else 0
        rows.append({"P": round(p, 2), "ES": round(es, 2)})
    oes = overall_exam_score([r["ES"] for r in rows], [it["V"] for it in items])
    acc_only = 100.0 * acc / len(items)
    return {
        "rows": rows,
        "oes": round(oes, 2),
        "accuracy_only_x100": round(acc_only, 2),
        "process_penalty_matters": abs(oes - acc_only) > 0.5,
    }
