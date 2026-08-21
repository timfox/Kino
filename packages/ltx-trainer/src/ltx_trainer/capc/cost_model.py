"""CAPC cost model: ρ_cross, two-tier ρ, strategies A–D (arXiv:2607.15516)."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.capc.config import CapcConfig


def rho_cross(r: float, cfg: CapcConfig | None = None) -> float:
    """Crossover cache-hit threshold (paper eqs. 5–6).

    ``ρ_cross(r) = (α − 1/r) / (α − β)`` with ``α = cw/pin``, ``β = cr/pin``.
    Cache-only (B) beats query-aware (C) only when empirical ρ exceeds this.
    """
    c = cfg or CapcConfig()
    r = max(1.0, float(r))
    alpha = c.alpha
    beta = c.beta
    denom = alpha - beta
    if denom <= 0:
        return 1.0
    raw = (alpha - 1.0 / r) / denom
    return float(max(0.0, min(1.0, raw)))


def rho_empirical(
    n_calls: int,
    prefix_tokens: int,
    cfg: CapcConfig | None = None,
) -> float:
    """Two-tier empirical hit rate ρ(N, |P|) (paper §3.1).

    Below ``T ≈ 3500``: hot tier plateau ``ρ_hot ≈ 0.83`` (climbs with N).
    At/above ``T``: persistent tier ``ρ ≈ 1.0`` from the first subsequent call.
    """
    c = cfg or CapcConfig()
    n = max(1, int(n_calls))
    p = max(0, int(prefix_tokens))
    t = max(1, int(c.hot_tier_tokens))
    if p >= t:
        return float(c.rho_persistent) if n >= 2 else float(c.rho_persistent)
    # Hot tier: climb toward plateau (paper: ~0.47 at N=5 → ~0.83 at N=30).
    plateau = float(c.rho_hot)
    if n <= 1:
        return 0.0
    # Smooth climb: 1 − exp(−(N−1)/12) scaled to plateau.
    climb = 1.0 - math.exp(-(n - 1) / 12.0)
    return float(min(plateau, plateau * climb / (1.0 - math.exp(-29.0 / 12.0))))


def tier_preserving_rmax(doc_tokens: int, cfg: CapcConfig | None = None) -> int:
    """Max integer ratio keeping compressed prefix in the *persistent* tier.

    Paper §5.2: ``r_safe_max(|D|) = ⌊|D|/T⌋`` so ``|D|/r ≥ T`` (avoid pushing
    the cached prefix *into* the hot tier below ~3500 tokens).
    """
    c = cfg or CapcConfig()
    d = max(1, int(doc_tokens))
    t = max(1, int(c.hot_tier_tokens))
    return max(1, d // t)


def tier_preserving_rmax_chars(doc_chars: int, max_chars: int, cfg: CapcConfig | None = None) -> int:
    """Char-budget analogue of ``tier_preserving_rmax`` for wake files."""
    _ = cfg
    d = max(1, int(doc_chars))
    budget = max(1, int(max_chars))
    return max(1, math.ceil(d / budget))


def _usd_per_tok(rate_per_mtok: float) -> float:
    return float(rate_per_mtok) / 1e6


def strategy_costs(
    *,
    prompt_tokens: int,
    doc_tokens: int,
    r: float,
    cfg: CapcConfig | None = None,
    n_calls: int = 10,
    output_tokens: int = 200,
    mutated_frac: float = 1.0,
) -> dict[str, Any]:
    """Per-query costs for strategies A–D (paper eqs. 1–4).

    A — Vanilla (no cache, no compression)
    B — Cache-only (full doc + cache_control)
    C — Query-aware compress (no stable prefix)
    D — CAPC (query-agnostic compress + cache)

    ``mutated_frac`` (§7.2): fraction of the cached prefix invalidated by
    query-aware compression (1.0 = full bust; small = tools= schema survives).
    """
    c = cfg or CapcConfig()
    dq = max(0, int(prompt_tokens))  # dynamic query block
    d = max(1, int(doc_tokens))
    r = max(1.0, float(r))
    n = max(1, int(n_calls))
    o = max(0, int(output_tokens))
    mf = float(max(0.0, min(1.0, mutated_frac)))

    compressed = max(1, int(round(d / r)))
    pin = _usd_per_tok(c.pin)
    cw = _usd_per_tok(c.cw)
    cr = _usd_per_tok(c.cr)
    pout = _usd_per_tok(c.pout)

    rho_b = rho_empirical(n, d, c)
    rho_d = rho_empirical(n, compressed, c)

    # A: full document every call, uncached.
    cost_a = (d + dq) * pin + o * pout

    # B: cache full document.
    cost_b = (1.0 - rho_b) * d * cw + rho_b * d * cr + dq * pin + o * pout

    # C: query-aware — compressed tokens at pin, but mutated_frac of an
    # otherwise-cacheable full-doc prefix pays write tax when mf>0.
    # When mf=1 (classic LongBench): pure pin on compressed (eq. 3).
    # When mf<1 (EA-style): residual cache hits on (1−mf)·|D|.
    if mf >= 1.0 - 1e-12:
        cost_c = (compressed + dq) * pin + o * pout
    else:
        # Unchanged fraction still caches at full-doc size; mutated region
        # is re-sent compressed at pin (no cache).
        stable = (1.0 - mf) * d
        rho_stable = rho_empirical(n, int(round(stable)), c) if stable >= 1 else 0.0
        cost_c = (
            (1.0 - rho_stable) * stable * cw
            + rho_stable * stable * cr
            + (mf * compressed + dq) * pin
            + o * pout
        )

    # D: CAPC — compressed once, cached.
    cost_d = (1.0 - rho_d) * compressed * cw + rho_d * compressed * cr + dq * pin + o * pout

    costs = {
        "A_vanilla": cost_a,
        "B_cache_only": cost_b,
        "C_query_aware": cost_c,
        "D_capc": cost_d,
    }
    ranking = sorted(costs.items(), key=lambda x: x[1])
    cheapest = ranking[0][0]
    return {
        "prompt_tokens": dq,
        "doc_tokens": d,
        "r": r,
        "n_calls": n,
        "output_tokens": o,
        "mutated_frac": mf,
        "compressed_tokens": compressed,
        "rho_cross": rho_cross(r, c),
        "rho_b": rho_b,
        "rho_d": rho_d,
        "costs": costs,
        "cheapest": cheapest,
        "capc_wins": cheapest == "D_capc",
        # Back-compat aliases used by older callers/tests during transition.
        "A_full": cost_a,
        "B_query_aware": cost_c,
        "C_capc": cost_d,
        "D_naive": cost_b,
    }
