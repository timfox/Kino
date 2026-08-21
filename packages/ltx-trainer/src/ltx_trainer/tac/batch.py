"""Batch TaC / TaC-C evaluation over a corpus."""

from __future__ import annotations

from typing import Any

from ltx_trainer.tac.pipeline import TaCExample, tac_c_select_trace, tac_vanilla


def evaluate_corpus(
    examples: list[TaCExample],
    *,
    compression_ratio: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    vanilla_rows = []
    tac_c_rows = []
    for i, ex in enumerate(examples):
        v = tac_vanilla(ex, compression_ratio=compression_ratio)
        c = tac_c_select_trace(ex, compression_ratio=compression_ratio, seed=seed + i)
        vanilla_rows.append(
            {
                "id": ex.example_id,
                "em": v["em"],
                "f1": v["f1"],
                "utility": v["utility"],
            }
        )
        tac_c_rows.append(
            {
                "id": ex.example_id,
                "em": c["em"],
                "f1": c["f1"],
                "best_reward": c["best_reward"]["total"],
            }
        )

    def _mean(key: str, rows: list[dict]) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "compression_ratio": compression_ratio,
        "num_examples": len(examples),
        "tac_vanilla": {
            "mean_em": _mean("em", vanilla_rows),
            "mean_f1": _mean("f1", vanilla_rows),
            "rows": vanilla_rows,
        },
        "tac_c": {
            "mean_em": _mean("em", tac_c_rows),
            "mean_f1": _mean("f1", tac_c_rows),
            "rows": tac_c_rows,
        },
    }
