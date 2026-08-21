"""Toy detection signals bridging MIA and contamination (Sec. 2–4)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MiaScore:
    perplexity: float
    loss_margin: float
    predicted_member: bool


def toy_perplexity(text: str) -> float:
    """Synthetic perplexity — repetitive text → lower ppl (member-like)."""
    tokens = text.split()
    if not tokens:
        return 50.0
    uniq_ratio = len(set(tokens)) / len(tokens)
    return 4.0 + 32.0 * uniq_ratio


def membership_infer(
    text: str,
    *,
    member_ceiling: float = 12.0,
    reference_ppl: float | None = None,
) -> MiaScore:
    """Heuristic MIA: low perplexity ⇒ member-like (illustrative only)."""
    ppl = toy_perplexity(text)
    ref = reference_ppl if reference_ppl is not None else member_ceiling + 10.0
    margin = ref - ppl
    return MiaScore(
        perplexity=round(ppl, 3),
        loss_margin=round(margin, 3),
        predicted_member=ppl <= member_ceiling,
    )


def ngram_overlap_fraction(text: str, reference: str, n: int = 5) -> float:
    """Dataset-level contamination signal — n-gram overlap rate."""
    def ngrams(s: str) -> set[tuple[str, ...]]:
        toks = s.split()
        if len(toks) < n:
            return {tuple(toks)} if toks else set()
        return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}

    a, b = ngrams(text), ngrams(reference)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


def contamination_flag(overlap: float, threshold: float = 0.35) -> bool:
    return overlap >= threshold


def min_k_percentile_gap(member_losses: list[float], nonmember_losses: list[float]) -> float:
    """Min-K style gap — lower member losses ⇒ positive gap."""
    if not member_losses or not nonmember_losses:
        return 0.0
    k = max(1, len(member_losses) // 10)
    mem_sorted = sorted(member_losses)[:k]
    non_sorted = sorted(nonmember_losses)[:k]
    return float(sum(non_sorted) / k - sum(mem_sorted) / k)


def losses_from_texts(texts: list[str], *, member_like: bool) -> list[float]:
    base = 1.2 if member_like else 2.8
    return [base + 0.01 * len(t) + 0.1 * math.sin(len(t)) for t in texts]
