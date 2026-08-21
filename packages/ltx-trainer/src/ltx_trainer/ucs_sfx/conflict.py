"""Per-file conflict resolution (Sec. 2.1.2)."""

from __future__ import annotations

from collections import Counter

from ltx_trainer.ucs_sfx.cascade import UcsMatch


def resolve_file_categories(matches: list[UcsMatch]) -> tuple[str, str | None, bool]:
    """Return (category, subcategory, ambiguous). Unclassified if no matches."""
    valid = [m for m in matches if m.stage != "unclassified" and m.category]
    if not valid:
        return "", None, False

    # 1. Specificity filter — prefer subcategory-level matches
    max_spec = max(m.specificity for m in valid)
    filtered = [m for m in valid if m.specificity == max_spec] or valid

    cats = [m.category for m in filtered]
    if len(set(cats)) == 1:
        sub = next((m.subcategory for m in filtered if m.subcategory), None)
        return cats[0], sub, len(set(m.category for m in valid)) > 1

    # 2. Majority vote
    counts = Counter(cats)
    top_count = counts.most_common(1)[0][1]
    winners = [c for c, n in counts.items() if n == top_count]
    if len(winners) == 1:
        cat = winners[0]
        sub = next((m.subcategory for m in filtered if m.category == cat and m.subcategory), None)
        return cat, sub, True

    # 3. Positional priority — last tag wins
    for m in reversed(valid):
        if m.category in winners:
            return m.category, m.subcategory, True
    return valid[-1].category, valid[-1].subcategory, True
