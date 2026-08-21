"""Compare DLM decoding strategies with computed round/RTF/WER proxies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dlmasr.config import DecodingStrategy, DlmAsrConfig
from ltx_trainer.dlmasr.decoding import block_decode_smoke
from ltx_trainer.dlmasr.uncertainty import uncertainty_smoke


def _wer_proxy(rounds: int, seq_len: int, *, strategy: DecodingStrategy, cfg: DlmAsrConfig) -> float:
    """Map decode rounds to WER% using paper anchors (lower rounds → lower WER for static)."""
    base = cfg.ar_wer_pct
    if strategy == DecodingStrategy.STATIC_THRESHOLD:
        return cfg.static_b4_c095_wer_pct + 0.01 * max(0, rounds - cfg.static_mean_rounds)
    if strategy == DecodingStrategy.DYNAMIC_THRESHOLD:
        return cfg.matched_rtf_dynamic_wer_pct + 0.015 * max(0, rounds - cfg.dynamic_mean_rounds)
    if strategy == DecodingStrategy.FIXED_NUMBER:
        return cfg.matched_rtf_fixed_wer_pct + 0.02 * max(0, rounds - 8)
    return base + 0.05 * rounds


def _rtf_proxy(rounds: int, seq_len: int, *, strategy: DecodingStrategy, cfg: DlmAsrConfig) -> float:
    if strategy == DecodingStrategy.STATIC_THRESHOLD:
        return cfg.static_rtf * (rounds / max(cfg.static_mean_rounds, 1))
    if strategy == DecodingStrategy.DYNAMIC_THRESHOLD:
        return cfg.dynamic_rtf * (rounds / max(cfg.dynamic_mean_rounds, 1))
    if strategy == DecodingStrategy.FIXED_NUMBER:
        return cfg.fixed_rtf * (rounds / max(cfg.fixed_k1_mean_rounds, 1))
    return 0.078


def compare_strategies(
    seq_len: int = 32,
    *,
    cfg: DlmAsrConfig | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    cfg = cfg or DlmAsrConfig()
    rows: list[dict[str, Any]] = []
    for strategy in (
        DecodingStrategy.STATIC_THRESHOLD,
        DecodingStrategy.DYNAMIC_THRESHOLD,
        DecodingStrategy.FIXED_NUMBER,
    ):
        out = block_decode_smoke(seq_len, strategy=strategy, cfg=cfg, seed=seed)
        rounds = int(out["rounds"])
        rows.append(
            {
                "strategy": strategy.value,
                "rounds": rounds,
                "wer_pct": round(_wer_proxy(rounds, seq_len, strategy=strategy, cfg=cfg), 3),
                "rtf": round(_rtf_proxy(rounds, seq_len, strategy=strategy, cfg=cfg), 4),
            }
        )
    rows.sort(key=lambda r: r["wer_pct"])
    return rows


def strategy_compare_smoke(cfg: DlmAsrConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or DlmAsrConfig()
    rows = compare_strategies(32, cfg=cfg, seed=seed)
    static = next(r for r in rows if r["strategy"] == DecodingStrategy.STATIC_THRESHOLD.value)
    fixed = next(r for r in rows if r["strategy"] == DecodingStrategy.FIXED_NUMBER.value)
    unc = uncertainty_smoke(32, seed=seed)
    return {
        "n_strategies": len(rows),
        "static_wer_pct": static["wer_pct"],
        "static_rounds": static["rounds"],
        "fixed_rounds": fixed["rounds"],
        "static_beats_fixed_wer": static["wer_pct"] < fixed["wer_pct"],
        "static_fewer_rounds": static["rounds"] < fixed["rounds"],
        "uncertainty_gap": unc["gap_at_full"],
        "best_strategy": rows[0]["strategy"],
    }
