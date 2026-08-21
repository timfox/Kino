"""Hybrid evaluation protocol for ReceiptBench (Sec. 3.4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.receiptbench.config import ReceiptBenchConfig
from ltx_trainer.receiptbench.matching import list_field_similarity
from ltx_trainer.receiptbench.schema import ALL_FIELDS, FIELD_SPEC, fields_for_subtask
from ltx_trainer.receiptbench.similarity import levenshtein_ratio, normalize_string, parse_amount


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, list):
        return len(v) == 0
    return False


def field_similarity(
    pred: Any,
    gold: Any,
    field: str,
    *,
    cfg: ReceiptBenchConfig | None = None,
    semantic_judge: callable[[str, str, str], float] | None = None,
) -> float:
    """Similarity S(Pf, Gf) in [0, 1] for reward / evaluation."""
    cfg = cfg or ReceiptBenchConfig()
    _, metric = FIELD_SPEC[field]
    pe, ge = _is_empty(pred), _is_empty(gold)

    if metric == "list":
        pl = pred if isinstance(pred, list) else []
        gl = gold if isinstance(gold, list) else []
        return list_field_similarity(pl, gl, cfg=cfg, is_detail=(field == "detail"))

    if pe and ge:
        return 1.0

    if metric == "numeric":
        pa, ga = parse_amount(pred), parse_amount(gold)
        if pa is None and ga is None:
            return 1.0
        if pa is None or ga is None:
            # Paper: zero and empty string are equivalent
            if (pa == 0 or pe) and (ga == 0 or ge):
                return 1.0
            return 0.0
        if abs(pa - ga) < cfg.numeric_epsilon:
            return 1.0
        return 0.0

    if pe or ge:
        return 0.0

    ps, gs = str(pred), str(gold)
    if normalize_string(ps) == normalize_string(gs):
        return 1.0
    if metric == "semantic" and semantic_judge is not None:
        return max(0.0, min(1.0, semantic_judge(field, gs, ps)))
    if metric == "semantic":
        return levenshtein_ratio(ps, gs)
    return 1.0 if normalize_string(ps) == normalize_string(gs) else 0.0


def confusion_counts(
    pred: dict[str, Any],
    gold: dict[str, Any],
    *,
    cfg: ReceiptBenchConfig | None = None,
    threshold: float = 0.5,
) -> dict[str, int]:
    """TP/TN/FP/FN counts across all fields."""
    cfg = cfg or ReceiptBenchConfig()
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for field in ALL_FIELDS:
        s = field_similarity(pred.get(field), gold.get(field), field, cfg=cfg)
        pe = _is_empty(pred.get(field))
        ge = _is_empty(gold.get(field))
        if not ge and not pe and s >= threshold:
            counts["tp"] += 1
        elif ge and pe and s >= threshold:
            counts["tn"] += 1
        elif ge and not pe:
            counts["fp"] += 1
        elif not ge and pe:
            counts["fn"] += 1
        elif not ge and not pe and s < threshold:
            counts["fn"] += 1  # partial miss treated as FN for F1
    return counts


def f1_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> float:
    """F1 with explicit TN handling for empty fields (Sec. 3.4)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_receipt(
    pred: dict[str, Any],
    gold: dict[str, Any],
    *,
    cfg: ReceiptBenchConfig | None = None,
) -> dict[str, Any]:
    """Full evaluation: per-field, per-subtask, overall F1."""
    cfg = cfg or ReceiptBenchConfig()
    counts = confusion_counts(pred, gold, cfg=cfg)
    overall_f1 = f1_from_counts(**counts)
    subtask_f1: dict[str, float] = {}
    for sub in ("perception", "normalization", "reasoning", "structure"):
        sc = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for field in fields_for_subtask(sub):  # type: ignore[arg-type]
            s = field_similarity(pred.get(field), gold.get(field), field, cfg=cfg)
            pe = _is_empty(pred.get(field))
            ge = _is_empty(gold.get(field))
            if not ge and not pe and s >= 0.5:
                sc["tp"] += 1
            elif ge and pe:
                sc["tn"] += 1
            elif ge and not pe:
                sc["fp"] += 1
            else:
                sc["fn"] += 1
        subtask_f1[sub] = f1_from_counts(**sc)
    return {
        "overall_f1": overall_f1,
        "subtask_f1": subtask_f1,
        "counts": counts,
        "field_scores": {
            f: field_similarity(pred.get(f), gold.get(f), f, cfg=cfg) for f in ALL_FIELDS
        },
    }
