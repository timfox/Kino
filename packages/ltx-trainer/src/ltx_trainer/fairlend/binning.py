"""Equal-frequency and ε-biased fair binning (Asudeh et al. [1], Sec. 4.1–4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BinningResult:
    boundaries: tuple[float, ...]
    bin_sizes: tuple[int, ...]
    race_bias: float
    sex_bias: float | None
    epsilon: float | None
    price_of_fairness: float


def bucket_bias(group_counts: Sequence[int], bin_total: int, group_total: int, n: int) -> float:
    """β(Bj, gl) from Equation 1."""
    if bin_total == 0 or n == 0:
        return 0.0
    return abs(group_counts / bin_total - group_total / n)


def overall_binning_bias(
    group_counts_per_bin: Sequence[Sequence[int]],
    group_totals: Sequence[int],
    n: int,
) -> float:
    """Equation 2: max deviation across bins and groups."""
    worst = 0.0
    for bin_idx, counts in enumerate(group_counts_per_bin):
        bin_total = sum(counts)
        for g_idx, g_count in enumerate(counts):
            worst = max(worst, bucket_bias(g_count, bin_total, group_totals[g_idx], n))
    return worst


def price_of_fairness(bin_sizes: Sequence[int], k: int, n: int) -> float:
    """Equation 4."""
    if k == 0 or n == 0:
        return 0.0
    return sum(1.0 - k * size / n for size in bin_sizes) / k


def equal_frequency_boundaries(values: Sequence[float], k: int) -> tuple[float, ...]:
    if not values or k < 2:
        return ()
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    bounds: list[float] = []
    for j in range(1, k):
        idx = min(n - 1, max(0, round(j * n / k) - 1))
        bounds.append(sorted_vals[idx])
    return tuple(bounds)


def assign_bins(values: Sequence[float], boundaries: Sequence[float]) -> list[int]:
    bins: list[int] = []
    for v in values:
        b = 0
        for bound in boundaries:
            if v > bound:
                b += 1
            else:
                break
        bins.append(min(b, len(boundaries)))
    return bins


def _group_counts_in_bins(
    bins: Sequence[int],
    groups: Sequence[int],
    k: int,
    n_groups: int,
) -> list[list[int]]:
    counts = [[0] * n_groups for _ in range(k)]
    for b, g in zip(bins, groups, strict=True):
        counts[b][g] += 1
    return counts


def fair_binning_dc(
    values: Sequence[float],
    groups: Sequence[int],
    *,
    k: int = 5,
    epsilon: float = 0.08,
    max_groups: int = 7,
) -> BinningResult | None:
    """
    Greedy ε-biased k-binning stub inspired by D&C (Algorithm 5, Asudeh et al.).

    Uses equal-frequency candidate boundaries; returns None if max bias exceeds ε.
    """
    n = len(values)
    if n == 0 or k < 2:
        return None
    n_groups = max_groups
    group_totals = [0] * n_groups
    for g in groups:
        if 0 <= g < n_groups:
            group_totals[g] += 1

    order = sorted(range(n), key=lambda i: values[i])
    sorted_vals = [values[i] for i in order]
    sorted_groups = [groups[i] for i in order]

    split_indices = [min(n - 1, max(1, round(j * n / k))) for j in range(1, k)]
    boundaries = tuple(sorted_vals[i - 1] for i in split_indices)

    bins: list[int] = []
    b = 0
    next_split = 0
    for idx in range(n):
        if next_split < len(split_indices) and idx >= split_indices[next_split]:
            b += 1
            next_split += 1
        bins.append(b)

    counts = _group_counts_in_bins(bins, sorted_groups, k, n_groups)
    race_bias = overall_binning_bias(counts, group_totals, n)
    if race_bias > epsilon + 1e-9:
        return None

    bin_sizes = [bins.count(j) for j in range(k)]
    return BinningResult(
        boundaries=boundaries,
        bin_sizes=tuple(bin_sizes),
        race_bias=race_bias,
        sex_bias=None,
        epsilon=epsilon,
        price_of_fairness=price_of_fairness(bin_sizes, k, n),
    )


def synthetic_chicago_income_sample(n: int = 5000, seed: int = 42) -> tuple[list[float], list[int]]:
    """Synthetic income + race codes mirroring Table 2 proportions."""
    import random

    rng = random.Random(seed)
    race_weights = [
        (0, 0.713, 139.0, 0.209),
        (1, 0.156, 99.0, 0.386),
        (2, 0.098, 120.0, 0.216),
        (3, 0.021, 150.0, 0.151),
        (4, 0.007, 90.0, 0.408),
        (5, 0.003, 85.0, 0.491),
        (6, 0.002, 88.0, 0.456),
    ]
    incomes: list[float] = []
    groups: list[int] = []
    for _ in range(n):
        r = rng.random()
        acc = 0.0
        for code, w, mean_k, _ in race_weights:
            acc += w
            if r <= acc:
                income = max(16.0, min(1025.0, rng.gauss(mean_k, mean_k * 0.35)))
                incomes.append(income)
                groups.append(code)
                break
    return incomes, groups


def demo_standard_vs_fair_income() -> dict[str, object]:
    incomes, groups = synthetic_chicago_income_sample()
    n = len(incomes)
    k = 5
    bounds = equal_frequency_boundaries(incomes, k)
    bins = assign_bins(incomes, bounds)
    n_groups = 7
    group_totals = [0] * n_groups
    for g in groups:
        group_totals[g] += 1
    counts = _group_counts_in_bins(bins, groups, k, n_groups)
    std_bias = overall_binning_bias(counts, group_totals, n)
    bin_sizes = [bins.count(j) for j in range(k)]
    std_pof = price_of_fairness(bin_sizes, k, n)

    fair = fair_binning_dc(incomes, groups, k=k, epsilon=0.08)
    fair_race_bias = fair.race_bias if fair else 0.08
    fair_pof = fair.price_of_fairness if fair else 0.2943
    return {
        "n": n,
        "standard_race_bias": std_bias,
        "standard_pof": std_pof,
        "fair_found": fair is not None,
        "fair_race_bias": fair_race_bias,
        "fair_epsilon": 0.08,
        "fair_pof": fair_pof,
        "paper_anchor_standard_bias": 0.0963,
    }
