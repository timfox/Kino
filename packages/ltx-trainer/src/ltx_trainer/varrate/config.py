"""VarRate — training-free variable-rank KV coding (arXiv:2607.15498)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15498"
PAPER_TITLE = "VarRate: Training-Free Variable-Rank KV Cache Coding"
PAPER_SYSTEM = "VarRate"
PAPER_AUTHORS = "Esmat et al."
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
BENCHMARK = "LongBench + reuse / prefill-overhead vs SnapKV / KVzip"

# Paper defaults
KAPPA = 0.20
RMIN = 16
R_MAX = 1024
STRIDE_DELTA = 16
WINDOW_W = 64
WINDOW_WOBS = 64


@dataclass
class VarRateConfig:
    """Runtime knobs for the VarRate CPU stub."""

    kappa: float = KAPPA
    rmin: int = RMIN
    r_max: int = R_MAX
    stride: int = STRIDE_DELTA
    window: int = WINDOW_W
    window_obs: int = WINDOW_WOBS
