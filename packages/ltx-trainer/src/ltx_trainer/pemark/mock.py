"""PEMark Lehmer position-encoding smoke (arXiv:2605.21865)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pemark.config import PEMarkConfig
from ltx_trainer.pemark.position_encoding import int_to_watermark_bits, min_threshold_T_for_bits


def evaluation_smoke(cfg: PEMarkConfig | None = None) -> dict[str, Any]:
    c = cfg or PEMarkConfig()
    bits_len = 8
    t_min = min_threshold_T_for_bits(bits_len)
    bits = int_to_watermark_bits(42, bits_len)
    return {
        "paper": c.paper_arxiv,
        "min_T": t_min,
        "watermark_bits": bits,
    }
