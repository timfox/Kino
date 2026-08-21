"""Programmatic verifiers for AgenticVBench task families (Appendix H)."""

from __future__ import annotations

from typing import Sequence


def assembly_score(correct_slots: int, total_slots: int, candidates_per_slot: int) -> float:
    """Eq. (1): chance-corrected per-slot accuracy."""
    if total_slots <= 0 or candidates_per_slot <= 1:
        return 1.0 if correct_slots == total_slots else 0.0
    r = correct_slots / total_slots
    k = float(candidates_per_slot)
    return (r - 1.0 / k) / (1.0 - 1.0 / k)


def normalized_distance(pred_order: Sequence[int], true_order: Sequence[int]) -> float:
    """ND from Appendix H.2."""
    n = len(true_order)
    if n == 0:
        return 0.0
    pred_rank = {clip: i for i, clip in enumerate(pred_order)}
    disp = sum(abs(pred_rank.get(c, n) - i) for i, c in enumerate(true_order))
    denom = n * n // 2
    return disp / denom if denom else 0.0


def lis_ratio(pred_order: Sequence[int], true_order: Sequence[int]) -> float:
    """LIS score: longest increasing subsequence of true ranks in predicted order."""
    n = len(true_order)
    if n == 0:
        return 1.0
    true_rank = {c: i for i, c in enumerate(true_order)}
    seq = [true_rank[c] for c in pred_order if c in true_rank]
    if len(seq) != n:
        return 0.0
    tails: list[int] = []
    for x in seq:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(x)
        else:
            tails[lo] = x
    return len(tails) / n


def adjacent_fidelity(pred_order: Sequence[int], true_order: Sequence[int]) -> float:
    """ADJ: fraction of ground-truth adjacent pairs preserved in prediction."""
    n = len(true_order)
    if n <= 1:
        return 1.0
    adj_pairs = {(true_order[i], true_order[i + 1]) for i in range(n - 1)}
    hits = 0
    for i in range(len(pred_order) - 1):
        if (pred_order[i], pred_order[i + 1]) in adj_pairs:
            hits += 1
    return hits / (n - 1)


def sequencing_score(pred_order: Sequence[int], true_order: Sequence[int]) -> float:
    """Eq. (3): (1 - ND) * LIS * ADJ."""
    if len(pred_order) != len(true_order) or set(pred_order) != set(true_order):
        return 0.0
    nd = normalized_distance(pred_order, true_order)
    lis = lis_ratio(pred_order, true_order)
    adj = adjacent_fidelity(pred_order, true_order)
    return max(0.0, (1.0 - nd) * lis * adj)


def strict_sequence_match(pred_order: Sequence[int], true_order: Sequence[int]) -> bool:
    return list(pred_order) == list(true_order)


def repair_linear_reward(
    measurement_out: float,
    measurement_broken: float,
    measurement_golden: float,
    *,
    lower_is_better: bool = False,
) -> float:
    """Appendix H.1: clip linear interpolation between broken and golden."""
    denom = measurement_golden - measurement_broken
    if abs(denom) < 1e-12:
        return 1.0 if abs(measurement_out - measurement_golden) < 1e-9 else 0.0
    if lower_is_better:
        s = (measurement_broken - measurement_out) / (measurement_broken - measurement_golden)
    else:
        s = (measurement_out - measurement_broken) / denom
    return max(0.0, min(1.0, s))


def repair_window_reward(
    in_window_score: float,
    out_window_score: float,
    *,
    in_weight: float = 0.9,
) -> float:
    """Audio/visual defects: 0.9 * s_in + 0.1 * s_out."""
    return in_weight * max(0.0, min(1.0, in_window_score)) + (1.0 - in_weight) * max(
        0.0, min(1.0, out_window_score)
    )


def timeline_range_match(
    predicted_ranges: Sequence[tuple[float, float]],
    gold_ranges: Sequence[tuple[float, float]],
    tolerance_s: float,
) -> float:
    """Fraction of gold ranges matched within tolerance (Repair timeline defects)."""
    if not gold_ranges:
        return 1.0
    used: set[int] = set()
    matched = 0
    for gs, ge in gold_ranges:
        best_i = -1
        best_dist = float("inf")
        for i, (ps, pe) in enumerate(predicted_ranges):
            if i in used:
                continue
            dist = abs(ps - gs) + abs(pe - ge)
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_i >= 0:
            ps, pe = predicted_ranges[best_i]
            if abs(ps - gs) <= tolerance_s and abs(pe - ge) <= tolerance_s:
                matched += 1
                used.add(best_i)
    return matched / len(gold_ranges)


def repurpose_rubric_score(yes_items: int, total_items: int) -> float:
    """Normalized binary rubric score for Visual/Narrative/Sound pillars."""
    if total_items <= 0:
        return 0.0
    return yes_items / total_items
