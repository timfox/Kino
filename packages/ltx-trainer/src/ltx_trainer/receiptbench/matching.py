"""Hungarian matching for structured list fields (Sec. 3.4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.receiptbench.config import ReceiptBenchConfig
from ltx_trainer.receiptbench.similarity import composite_similarity, parse_amount


def _item_content(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(item)


def _item_amount(item: Any) -> float | None:
    if isinstance(item, dict):
        return parse_amount(item.get("amount"))
    return None


def _greedy_assign(cost: list[list[float]]) -> list[int]:
    """Greedy minimum-cost assignment (stable for small receipt lists)."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0]) if cost else 0
    used_cols: set[int] = set()
    assign = [-1] * n
    for i in range(n):
        best_j, best_c = -1, float("inf")
        for j in range(m):
            if j in used_cols:
                continue
            c = cost[i][j]
            if c < best_c:
                best_c, best_j = c, j
        if best_j >= 0:
            assign[i] = best_j
            used_cols.add(best_j)
    return assign


def list_field_similarity(
    pred: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    cfg: ReceiptBenchConfig | None = None,
    is_detail: bool = True,
) -> float:
    """Maximum bipartite matching score for list fields."""
    cfg = cfg or ReceiptBenchConfig()
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    n, m = len(pred), len(gold)
    size = max(n, m)
    cost = [[1.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            sim = composite_similarity(
                _item_content(pred[i]),
                _item_content(gold[j]),
                alpha=cfg.weight_levenshtein,
                beta=cfg.weight_token_sort,
                gamma=cfg.weight_lcs,
                delta=cfg.weight_semantic,
            )
            if is_detail:
                pa, ga = _item_amount(pred[i]), _item_amount(gold[j])
                if pa is not None and ga is not None and abs(pa - ga) > cfg.amount_mismatch_cost:
                    cost[i][j] = 1e9
                    continue
            cost[i][j] = 1.0 - sim
    assign = _greedy_assign(cost)
    matched = 0
    sim_sum = 0.0
    for i, j in enumerate(assign):
        if j < 0 or j >= m:
            continue
        c = cost[i][j]
        if c >= 1e8:
            continue
        if (1.0 - c) >= (1.0 - cfg.list_match_threshold):
            matched += 1
            sim_sum += 1.0 - c
    denom = max(n, m)
    return sim_sum / denom if denom else 1.0
