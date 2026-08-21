"""Lexicon evaluation metrics — NES, WNES, PAcc + inverses (arXiv:2606.06183)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def levenshtein(a: str, b: str) -> int:
    """Edit distance on whitespace-tokenized phoneme strings."""
    ta = a.split()
    tb = b.split()
    if not ta:
        return len(tb)
    if not tb:
        return len(ta)
    prev = list(range(len(tb) + 1))
    for i, ca in enumerate(ta, 1):
        cur = [i]
        for j, cb in enumerate(tb, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def normalized_edit_distance(t: str, t_prime: str) -> float:
    """Eq. (2) — lower is better."""
    denom = max(len(t.split()), len(t_prime.split()), 1)
    return levenshtein(t, t_prime) / denom


def normalized_edit_similarity(t: str, t_prime: str) -> float:
    return 1.0 - normalized_edit_distance(t, t_prime)


def _pairwise_ned_mean(units: Sequence[str], *, ignore_singleton: bool) -> float | None:
    n = len(units)
    if n < 2:
        return 0.0 if not ignore_singleton and n == 1 else None
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += normalized_edit_distance(units[i], units[j])
            pairs += 1
    return total / pairs if pairs else None


def nes(clusters: Mapping[int, Sequence[str]]) -> float:
    """Standard NES (1 − NED), Eq. (3) complement — ignores singleton clusters."""
    num = 0.0
    den = 0
    for units in clusters.values():
        if len(units) < 2:
            continue
        pairs = len(units) * (len(units) - 1) // 2
        mean_ned = _pairwise_ned_mean(units, ignore_singleton=False)
        if mean_ned is None:
            continue
        num += mean_ned * pairs
        den += pairs
    if den == 0:
        return 1.0
    return 1.0 - num / den


def wnes(clusters: Mapping[int, Sequence[str]]) -> float:
    """Weighted NES, Eq. (5) — each unit contributes equally."""
    weighted_sum = 0.0
    total_units = 0
    for units in clusters.values():
        n = len(units)
        if n == 0:
            continue
        if n == 1:
            # singleton clusters credited with zero error (§IV-C)
            total_units += 1
            continue
        mean_ned = _pairwise_ned_mean(units, ignore_singleton=False) or 0.0
        weighted_sum += n * mean_ned
        total_units += n
    if total_units == 0:
        return 1.0
    return 1.0 - weighted_sum / total_units


def pacc(clusters: Mapping[int, Sequence[str]]) -> float:
    """Phoneme accuracy, Eq. (6)."""
    if not clusters:
        return 1.0
    err = 0.0
    count = 0
    for units in clusters.values():
        if not units:
            continue
        if len(units) == 1:
            count += 1
            continue
        modal = max(units, key=units.count)
        denom = max(len(modal.split()), 1)
        for t in units:
            err += levenshtein(t, modal) / denom
            count += 1
    if count == 0:
        return 1.0
    return 1.0 - err / count


def iwnes(classes: Mapping[int, Sequence[str]]) -> float:
    """Inverse WNES, Eq. (7) — ignores singleton classes."""
    weighted_sum = 0.0
    total = 0
    for seqs in classes.values():
        n = len(seqs)
        if n < 2:
            continue
        mean_ned = _pairwise_ned_mean([str(s) for s in seqs], ignore_singleton=False) or 0.0
        weighted_sum += n * mean_ned
        total += n
    if total == 0:
        return 1.0
    return 1.0 - weighted_sum / total


def ipacc(classes: Mapping[int, Sequence[str]]) -> float:
    """Inverse PAcc, Eq. (8)."""
    err = 0.0
    count = 0
    for seqs in classes.values():
        if len(seqs) < 2:
            continue
        str_seqs = [str(s) for s in seqs]
        modal = max(str_seqs, key=str_seqs.count)
        denom = max(len(modal.split()), 1)
        for y in str_seqs:
            err += levenshtein(y, modal) / denom
            count += 1
    if count == 0:
        return 1.0
    return 1.0 - err / count


def harmonic_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def f1_wnes(forward: float, inverse: float) -> float:
    return harmonic_mean(forward, inverse)


def d_pacc(forward: float, inverse: float) -> float:
    """Eq. (9) — complement of Euclidean distance from (1, 1)."""
    return 1.0 - math.sqrt((1.0 - forward) ** 2 + (1.0 - inverse) ** 2)


def bitrate(
    cluster_sizes: Sequence[int],
    *,
    num_units: int,
    duration_s: float,
) -> float:
    """Entropic bitrate, Eq. (4) — lower is better (bits/s)."""
    if duration_s <= 0 or num_units <= 0:
        return 0.0
    bits = 0.0
    for size in cluster_sizes:
        if size <= 0:
            continue
        p = size / num_units
        bits -= size * math.log2(p)
    return bits / duration_s


def evaluate_lexicon(
    clusters: Mapping[int, Sequence[str]],
    classes: Mapping[int, Sequence[str]],
) -> dict[str, float]:
    fwd_nes = nes(clusters)
    fwd_wnes = wnes(clusters)
    fwd_pacc = pacc(clusters)
    inv_ines = iwnes({k: v for k, v in classes.items()})  # unweighted inverse of NES uses iWNES form
    inv_wnes = iwnes(classes)
    inv_pacc = ipacc(classes)
    return {
        "NES": fwd_nes,
        "WNES": fwd_wnes,
        "PAcc": fwd_pacc,
        "iWNES": inv_wnes,
        "iPAcc": inv_pacc,
        "F1-WNES": f1_wnes(fwd_wnes, inv_wnes),
        "F1-NES": f1_wnes(fwd_nes, inv_ines),
        "d-PAcc": d_pacc(fwd_pacc, inv_pacc),
    }


def toy_lexicon_demo() -> dict[str, object]:
    """Minimal lexicon illustrating NES cluster-size bias vs WNES fairness."""
    # Large cluster 0 dominates NES; small pure clusters lift WNES for impure lexicon.
    large_impure = {
        0: ["DH AH", "K AE T", "F IH SH D", "DH AH K", "K AE T S", "F IH SH", "DH AH", "K AE T"],
        1: ["DH AH", "DH AH"],
        2: ["K AE T", "K AE T"],
    }
    large_pure = {
        0: ["DH AH", "DH AH", "DH AH", "DH AH", "DH AH", "DH AH", "DH AH", "DH AH"],
        1: ["K AE T", "F IH SH D", "K AE T S", "DH AH K"],
        2: ["K AE T", "K AE T"],
    }
    classes = {
        0: ["0", "0", "0", "0"],
        1: ["1", "2", "1", "2"],
        2: ["2", "2"],
    }
    impure_scores = evaluate_lexicon(large_impure, classes)
    pure_scores = evaluate_lexicon(large_pure, classes)
    return {
        "large_impure": impure_scores,
        "large_pure": pure_scores,
        "wnes_prefers_impure_paper": impure_scores["WNES"] > pure_scores["WNES"],
        "nes_skews_pure": pure_scores["NES"] > impure_scores["NES"],
        "f1_wnes_balanced": abs(impure_scores["F1-WNES"] - pure_scores["F1-WNES"]) < 0.2,
    }
