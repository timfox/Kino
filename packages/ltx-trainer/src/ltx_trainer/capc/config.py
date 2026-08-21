"""CAPC — Cache-Aware Prompt Compression (arXiv:2607.15516)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15516"
PAPER_TITLE = "Cache-Aware Prompt Compression: A Two-Tier Cost Model for LLM API Caching"
PAPER_SYSTEM = "CAPC"
PAPER_AUTHORS = "Song et al."
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
BENCHMARK = "LongBench-v2 (16/16) + τ-bench retail"

# Persistent / hot tier boundary (Sonnet 4.6 May 2026 snapshot).
HOT_TIER_TOKENS = 3500
RHO_HOT = 0.83
RHO_PERSISTENT = 1.0

# Sonnet 4.6 $/MTok (paper §3.3): uncached input, cache write (5-min), cache read, output.
PIN_USD_PER_MTOK = 3.00
CW_USD_PER_MTOK = 3.75
CR_USD_PER_MTOK = 0.30
POUT_USD_PER_MTOK = 15.00


@dataclass
class CapcConfig:
    """Runtime knobs for the CAPC CPU stub."""

    hot_tier_tokens: int = HOT_TIER_TOKENS
    rho_hot: float = RHO_HOT
    rho_persistent: float = RHO_PERSISTENT
    pin: float = PIN_USD_PER_MTOK
    cw: float = CW_USD_PER_MTOK
    cr: float = CR_USD_PER_MTOK
    pout: float = POUT_USD_PER_MTOK
    # Approx chars per token for wake / char-budget compressors.
    chars_per_token: float = 4.0
    # AdaptiveCacheBoundary thresholds (paper §5.3).
    eps_static: float = 0.05
    eps_quasi: float = 0.30
    min_calls: int = 3

    @property
    def alpha(self) -> float:
        """Write-rate premium α = cw / pin."""
        return float(self.cw) / max(1e-12, float(self.pin))

    @property
    def beta(self) -> float:
        """Read-rate discount β = cr / pin."""
        return float(self.cr) / max(1e-12, float(self.pin))
