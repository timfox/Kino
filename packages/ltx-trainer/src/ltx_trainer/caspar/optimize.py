"""Symbolic optimizations: CSE, partial CSE, context-aware rewrites."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations


@dataclass
class SumExpr:
    terms: list[str]


def parse_sum(expr: str) -> SumExpr:
    parts = [p.strip() for p in expr.split("+")]
    return SumExpr(terms=parts)


def format_sum(terms: list[str]) -> str:
    return "+".join(terms)


def _replace_subset(terms: list[str], subset: frozenset[str], temp: str) -> list[str] | None:
    if not subset.issubset(terms):
        return None
    remaining = [t for t in terms if t not in subset]
    return [temp, *remaining]


def common_partial_cse(sums: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Paper §II-B partial subexpression elimination on commutative sums.

    Example::
        r0=a+b+c, r1=a+c+d+e, r2=a+c+e  →  r0=r3+b, r1=r2+d, r2=r3+e, r3=a+c
    """
    result_terms = [list(parse_sum(s).terms) for s in sums]
    temps: dict[str, str] = {}
    result_names = [f"r{i}" for i in range(len(sums))]

    while True:
        subset_counts: Counter[frozenset[str]] = Counter()
        for terms in result_terms:
            seen: set[frozenset[str]] = set()
            uniq = frozenset(terms)
            for size in range(2, len(uniq) + 1):
                for combo in combinations(uniq, size):
                    sub = frozenset(combo)
                    if sub not in seen:
                        seen.add(sub)
                        subset_counts[sub] += 1

        candidates = [(c, sub) for sub, c in subset_counts.items() if c > 1]
        if not candidates:
            break
        candidates.sort(key=lambda x: (-x[0], -len(x[1])))
        sub = candidates[0][1]
        temp_name = f"r{len(temps) + len(sums)}"
        temps[temp_name] = format_sum(sorted(sub, key=lambda t: terms.index(t) if (terms := list(sub)) else t))

        # Preserve first-seen term order for the temp definition.
        for orig in result_terms:
            if sub.issubset(orig):
                temps[temp_name] = "+".join(t for t in orig if t in sub)
                break

        new_terms: list[list[str]] = []
        for terms in result_terms:
            replaced = _replace_subset(terms, sub, temp_name)
            new_terms.append(replaced if replaced is not None else terms)
        result_terms = new_terms

        # Reuse an existing result row when it equals temp+single term (paper step 2).
        for idx, terms in enumerate(result_terms):
            if len(terms) == 2 and terms[0] == temp_name:
                alias = result_names[idx]
                for j, other in enumerate(result_terms):
                    if j == idx:
                        continue
                    if temp_name in other and len(other) == len(sub) + 1:
                        rest = [t for t in other if t != temp_name]
                        if len(rest) == 1:
                            result_terms[j] = [alias, *rest]

    out = [format_sum(t) for t in result_terms]
    return out, temps


def context_aware_reciprocal_mul(exprs: list[str]) -> list[str]:
    """Replace paired divisions a/x, b/x with r=1/x; r*a; r*b (§II-C)."""
    denom_re = re.compile(r"^(.+)/(.+)$")
    by_denom: dict[str, list[tuple[int, str]]] = {}
    for i, e in enumerate(exprs):
        m = denom_re.match(e.replace(" ", ""))
        if m:
            num, den = m.group(1), m.group(2)
            by_denom.setdefault(den, []).append((i, num))

    rewritten = list(exprs)
    for den, pairs in by_denom.items():
        if len(pairs) < 2:
            continue
        r = f"rcp({den})"
        for idx, num in pairs:
            rewritten[idx] = f"{r}*{num}"
    return rewritten


def hardware_map_norm(expr: str) -> str:
    """Map norm3/rnorm3 to hardware instruction names (stub labels)."""
    if "norm(" in expr and expr.count(",") == 0:
        return expr.replace("norm(", "norm3(")
    if "normalize(" in expr:
        return expr.replace("normalize(", "rnorm3(")
    return expr


def count_add_instructions(exprs: list[str]) -> int:
    """Count binary additions in flattened sum expressions."""
    total = 0
    for e in exprs:
        total += max(0, e.count("+"))
    return total
