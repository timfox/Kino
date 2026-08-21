"""Evaluation metrics: CSA, CRA, VSC, SQ, WCD, MSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class MetricBundle:
    csa: float
    cra: float
    vsc: float
    sq: float
    wcd: float
    mss: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "CSA": self.csa,
            "CRA": self.cra,
            "VSC": self.vsc,
            "SQ": self.sq,
            "WCD": self.wcd,
            "MSS": self.mss,
        }


def clips_selection_accuracy(
    predicted_positive: Sequence[bool],
    ground_truth_positive: Sequence[bool],
) -> float:
    """CSA: fraction of samples where only positive clips are selected."""
    if not predicted_positive:
        return 0.0
    # sample-level: all selected must be true positives in GT set
    selected = [i for i, p in enumerate(predicted_positive) if p]
    if not selected:
        return 0.0
    return float(all(ground_truth_positive[i] for i in selected))


def clips_rank_accuracy(predicted_order: Sequence[int], reference_order: Sequence[int]) -> float:
    """CRA: exact match on clip ordering."""
    return 1.0 if list(predicted_order) == list(reference_order) else 0.0


def visual_script_correlation_score(
    frame_caption: str,
    script_line: str,
    *,
    keyword_overlap_weight: float = 0.5,
) -> int:
    """Heuristic VSC in {0,1,2} for smoke (GPT-4o in paper)."""
    a = set(frame_caption.lower().split())
    b = set(script_line.lower().split())
    if not a or not b:
        return 0
    overlap = len(a & b) / max(len(a | b), 1)
    if overlap >= 0.35:
        return 2
    if overlap >= 0.12:
        return 1
    return 0


def script_quality_heuristic(
    generated: str,
    reference: str,
    *,
    product_keywords: Sequence[str] | None = None,
) -> float:
    """SQ proxy on [0, 100] using length + keyword coverage."""
    gen_words = generated.split()
    ref_words = reference.split()
    if not gen_words:
        return 0.0
    len_score = max(0.0, 30.0 - abs(len(gen_words) - len(ref_words)) * 2.0)
    overlap = len(set(gen_words) & set(ref_words)) / max(len(set(ref_words)), 1)
    expr_score = min(40.0, overlap * 50.0)
    kw_score = 30.0
    if product_keywords:
        hits = sum(1 for k in product_keywords if k.lower() in generated.lower())
        kw_score = min(30.0, hits / max(len(product_keywords), 1) * 30.0)
    return min(100.0, len_score + expr_score + kw_score)


def word_count_discrepancy(script: str, target_words: int) -> float:
    """WCD = |words(script) - target|."""
    return abs(len(script.split()) - target_words)


def music_similarity_score(
    pred_description: str,
    gt_description: str,
) -> float:
    """MSS proxy in [0,1] via description token overlap."""
    a = set(pred_description.lower().split())
    b = set(gt_description.lower().split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def aggregate_metrics(rows: Sequence[MetricBundle]) -> MetricBundle:
    if not rows:
        return MetricBundle(0, 0, 0, 0, 0, 0)
    n = len(rows)
    mss_vals = [r.mss for r in rows if r.mss is not None]
    return MetricBundle(
        csa=sum(r.csa for r in rows) / n,
        cra=sum(r.cra for r in rows) / n,
        vsc=sum(r.vsc for r in rows) / n,
        sq=sum(r.sq for r in rows) / n,
        wcd=sum(r.wcd for r in rows) / n,
        mss=(sum(mss_vals) / len(mss_vals)) if mss_vals else None,
    )
