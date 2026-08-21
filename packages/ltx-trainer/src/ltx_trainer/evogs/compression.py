"""Inter-layer ψ compression (Sec. 4.3, Table 5)."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from ltx_trainer.evogs.metrics import psnr


def uniform_quantize(x: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float, float]:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return x.copy(), lo, hi
    levels = (1 << bits) - 1
    q = np.round((x - lo) / (hi - lo) * levels) / levels * (hi - lo) + lo
    return q.astype(np.float64), lo, hi


def compress_psi_block(psi: np.ndarray) -> dict[str, Any]:
    q, lo, hi = uniform_quantize(psi, bits=8)
    payload = q.astype(np.float32).tobytes() + np.array([lo, hi], dtype=np.float32).tobytes()
    compressed = zlib.compress(payload, level=9)
    return {
        "raw_bytes": len(payload),
        "compressed_bytes": len(compressed),
        "ratio": len(payload) / max(len(compressed), 1),
        "dequant": q,
    }


def compress_tree_refinements(psis: list[np.ndarray]) -> dict[str, Any]:
    if not psis:
        return {
            "num_blocks": 0,
            "raw_mb": 0.0,
            "compressed_mb": 0.0,
            "compression_ratio": 1.0,
            "mean_abs_quant_err": 0.0,
        }
    blocks = [compress_psi_block(p) for p in psis]
    raw = sum(b["raw_bytes"] for b in blocks)
    comp = sum(b["compressed_bytes"] for b in blocks)
    recon = [b["dequant"] for b in blocks]
    err = float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(psis, recon, strict=False)])) if psis else 0.0
    return {
        "num_blocks": len(blocks),
        "raw_mb": raw / 1e6,
        "compressed_mb": comp / 1e6,
        "compression_ratio": raw / max(comp, 1),
        "mean_abs_quant_err": err,
    }


def compressed_storage_table_row(uncompressed_mb: float, *, ratio: float = 3.8) -> float:
    """Map uncompressed EvoGS storage to w/ Comp. column (Table 5)."""
    return uncompressed_mb / ratio
