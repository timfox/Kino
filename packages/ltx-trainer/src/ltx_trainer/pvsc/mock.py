"""PVSC entropy rate-matching smoke (arXiv:2605.19397)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pvsc.config import PVSCConfig
from ltx_trainer.pvsc.entropy import quantize_rate, symbol_length_factor


def evaluation_smoke(cfg: PVSCConfig | None = None) -> dict[str, Any]:
    c = cfg or PVSCConfig()
    k = symbol_length_factor(0.5, eta=0.2, channels=128)
    rate = quantize_rate(k)
    return {
        "paper": c.paper_arxiv,
        "symbol_length_k": k,
        "quantized_rate": rate,
    }
