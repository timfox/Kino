"""Evaluation metrics for MTAVG-Bench 2.0 (Appendix E.3–E.4)."""

from __future__ import annotations

from typing import Iterable


def score_single_choice(predicted: str, ground_truth: str) -> float:
    """Eq. (1): exact match for MCQ single and pairwise."""
    return 1.0 if str(predicted).strip() == str(ground_truth).strip() else 0.0


def score_multiple_choice(predicted: Iterable[str], ground_truth: Iterable[str]) -> float:
    """Eq. (2): fraction of ground-truth options covered by prediction."""
    gt = {str(x).strip() for x in ground_truth}
    if not gt:
        return 0.0
    pred = {str(x).strip() for x in predicted}
    return len(pred & gt) / len(gt)


def score_pairwise(predicted: str, ground_truth: str) -> float:
    """Eq. (3): same as single-choice."""
    return score_single_choice(predicted, ground_truth)


def sub_dimension_score(per_question_scores: list[float]) -> float:
    """Eq. (4): mean over questions in sub-dimension."""
    if not per_question_scores:
        return float("nan")
    return sum(per_question_scores) / len(per_question_scores)


def weighted_average_sub_dims(
    sub_dim_scores: dict[str, float],
    sub_dim_counts: dict[str, int],
) -> float:
    """Eq. (5): weighted mean over ten sub-dimensions."""
    total_n = sum(sub_dim_counts.values())
    if total_n == 0:
        return float("nan")
    acc = 0.0
    for code, score in sub_dim_scores.items():
        n = sub_dim_counts.get(code, 0)
        acc += (n / total_n) * score
    return acc


def primary_issue_accuracy(predicted_labels: list[str], ground_truth_labels: list[str]) -> float:
    """Eq. (6): PIA for temporal localization."""
    if len(predicted_labels) != len(ground_truth_labels) or not predicted_labels:
        return float("nan")
    hits = sum(1 for p, g in zip(predicted_labels, ground_truth_labels, strict=True) if p == g)
    return hits / len(predicted_labels)


def temporal_localization_accuracy(
    predicted_regions: list[set[str]],
    ground_truth_regions: list[set[str]],
) -> float:
    """Eq. (7): TLA — coverage of annotated failure time bins."""
    if len(predicted_regions) != len(ground_truth_regions) or not predicted_regions:
        return float("nan")
    scores: list[float] = []
    for pred, gt in zip(predicted_regions, ground_truth_regions, strict=True):
        if not gt:
            scores.append(0.0)
        else:
            scores.append(len(pred & gt) / len(gt))
    return sum(scores) / len(scores)


def rationale_consistency_pct(likert_scores: list[float]) -> float:
    """Eq. (8): RC on 1–5 Likert, reported as percentage."""
    if not likert_scores:
        return float("nan")
    return 100.0 * (sum(likert_scores) / (5.0 * len(likert_scores)))


def failure_rate(failure_segments: int, total_videos: int) -> float:
    """Eq. (9): clip-segment-level failure incidence FR_{m,d}."""
    if total_videos <= 0:
        return float("nan")
    return failure_segments / total_videos


def holistic_quality_index(
    *,
    audio_aesthetic: float,
    lip_sync: float,
    av_align: float,
    desync: float,
    ta_align: float,
    tv_align: float,
    desync_lower_is_better: bool = True,
) -> float:
    """Toy composite from Table 4 metrics (higher = better); not in paper — for LTX ranking smoke."""
    desync_term = 1.0 - min(desync, 1.0) if desync_lower_is_better else min(desync, 1.0)
    parts = [
        audio_aesthetic / 5.0,
        lip_sync,
        av_align,
        desync_term,
        ta_align,
        tv_align,
    ]
    return float(sum(parts) / len(parts))


def parse_mcq_answer(raw: str, *, valid_options: Iterable[str] | None = None) -> str | None:
    """Normalize model output to a single option id (stub parser)."""
    text = raw.strip()
    if not text:
        return None
    if valid_options is not None:
        opts = {str(o).strip() for o in valid_options}
        for line in text.splitlines():
            line = line.strip()
            if line in opts:
                return line
        for opt in opts:
            if opt in text:
                return opt
    return text.split()[0] if text.split() else None


def parse_multi_answer(raw: str) -> list[str]:
    """Extract comma- or newline-separated multi-label answer."""
    parts: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    if not parts and raw.strip():
        parts = [raw.strip()]
    return parts


def likert_to_binary_pass(score: float, *, threshold: float = 3.0) -> bool:
    return score >= threshold
