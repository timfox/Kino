"""Stance simulation metrics (arXiv:2606.06443)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

STANCE_ORDINAL: dict[str, int] = {"negative": -1, "neutral": 0, "positive": 1}


def stance_score(label: str) -> int:
    """Map three-way stance label to ordinal score (−1, 0, +1)."""
    key = label.strip().lower()
    if key not in STANCE_ORDINAL:
        raise ValueError(f"unknown stance label: {label!r}")
    return STANCE_ORDINAL[key]


def directional_stance_shift(inferred: str, revised: str) -> float:
    """Average directional shift Δ = score(revised) − score(inferred)."""
    return float(stance_score(revised) - stance_score(inferred))


def average_directional_shift(pairs: Sequence[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(directional_stance_shift(a, b) for a, b in pairs) / len(pairs)


def pct_change_from_baseline(baseline_mean: float, delta: float) -> float:
    """Percent change relative to mean inferred stance score (paper Table 2)."""
    if abs(baseline_mean) < 1e-9:
        return 0.0
    return 100.0 * delta / baseline_mean


def transition_rate(
    inferred: Sequence[str],
    revised: Sequence[str],
    *,
    from_label: str,
    to_label: str,
) -> float:
    """Fraction of instances with y_infer=from_label and y_rev=to_label."""
    if len(inferred) != len(revised):
        raise ValueError("inferred and revised must have equal length")
    num = sum(1 for a, b in zip(inferred, revised, strict=True) if a == from_label and b == to_label)
    denom = sum(1 for a in inferred if a == from_label)
    return num / denom if denom else 0.0


def supportive_transition_rate(inferred: Sequence[str], revised: Sequence[str]) -> float:
    """Combined neg→{neu,pos} and neu→pos transitions (paper Eq. 7 supportive branch)."""
    if len(inferred) != len(revised):
        raise ValueError("inferred and revised must have equal length")
    supportive = 0
    eligible = 0
    for y0, y1 in zip(inferred, revised, strict=True):
        if y0 == "negative":
            eligible += 1
            if y1 in ("neutral", "positive"):
                supportive += 1
        elif y0 == "neutral":
            eligible += 1
            if y1 == "positive":
                supportive += 1
    return supportive / eligible if eligible else 0.0


def backfire_rate(inferred: Sequence[str], revised: Sequence[str]) -> float:
    """Neutral→negative transitions (Nyhan–Reifler backfire proxy)."""
    return transition_rate(inferred, revised, from_label="neutral", to_label="negative")


def classification_metrics(
    observed: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, float]:
    """Accuracy, macro-F1, weighted-F1 for three-way stance (Stage 1 validation)."""
    if len(observed) != len(predicted) or not observed:
        return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    labels = ("negative", "neutral", "positive")
    correct = sum(1 for o, p in zip(observed, predicted, strict=True) if o == p)
    accuracy = correct / len(observed)

    per_label_f1: list[float] = []
    weights: list[int] = []
    for lab in labels:
        tp = sum(1 for o, p in zip(observed, predicted, strict=True) if o == lab and p == lab)
        fp = sum(1 for o, p in zip(observed, predicted, strict=True) if o != lab and p == lab)
        fn = sum(1 for o, p in zip(observed, predicted, strict=True) if o == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label_f1.append(f1)
        weights.append(sum(1 for o in observed if o == lab))

    macro_f1 = sum(per_label_f1) / len(labels)
    total = sum(weights) or 1
    weighted_f1 = sum(f * w for f, w in zip(per_label_f1, weights, strict=True)) / total
    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
    }


def depolarization_rate(
    original_tones: Sequence[float],
    revised_tones: Sequence[float],
    *,
    midpoint: float = 50.0,
) -> dict[str, float]:
    """LIWC Tone depolarization: low→high and high→low shift rates (Table 5)."""
    if len(original_tones) != len(revised_tones):
        raise ValueError("tone sequences must match")
    low_to_high = 0
    low_total = 0
    high_to_low = 0
    high_total = 0
    for o, r in zip(original_tones, revised_tones, strict=True):
        if o <= midpoint:
            low_total += 1
            if r > o:
                low_to_high += 1
        else:
            high_total += 1
            if r < o:
                high_to_low += 1
    l2h = 100.0 * low_to_high / low_total if low_total else 0.0
    h2l = 100.0 * high_to_low / high_total if high_total else 0.0
    overall = (l2h + h2l) / 2.0
    return {
        "low_to_high_pct": round(l2h, 1),
        "high_to_low_pct": round(h2l, 1),
        "overall_depolarize_pct": round(overall, 1),
    }


def macro_f1_from_counts(counts: Mapping[str, Mapping[str, float]]) -> float:
    """Macro-F1 from per-target accuracy/macro_f1 rows when only macro_f1 is stored."""
    vals = [float(row.get("macro_f1", 0.0)) for row in counts.values()]
    return sum(vals) / len(vals) if vals else 0.0


def mean_delta_by_strategy(
    rows: Iterable[Mapping[str, float | str]],
    *,
    strategy: str,
) -> float:
    deltas = [float(r[strategy]) for r in rows if strategy in r]
    return sum(deltas) / len(deltas) if deltas else 0.0
