"""Relay information bottleneck and MAS gain (Theorem 4.1)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.mas_ib.config import MasIbConfig


@dataclass
class RelayStats:
    """Stage-wise relay compression quantities."""

    context_entropy: float  # H(M_i)
    relay_entropy: float  # H(m_i) ≤ B_sys
    context_reduction: float  # H(M_i | m_i) ≈ H(M) − H(m) under deterministic compression proxy
    info_loss: float  # Δ_i(m)
    beta: float
    mas_gain: float  # H(M|m) − β Δ

    def as_dict(self) -> dict:
        return {
            "H_M": round(self.context_entropy, 4),
            "H_m": round(self.relay_entropy, 4),
            "context_reduction": round(self.context_reduction, 4),
            "delta_loss": round(self.info_loss, 4),
            "beta": round(self.beta, 4),
            "mas_gain": round(self.mas_gain, 4),
            "helps": self.mas_gain > 0,
        }


def context_reduction(full_bits: float, relay_bits: float) -> float:
    """Proxy for H(M|m): bits removed by compression (non-negative)."""
    return max(0.0, full_bits - relay_bits)


def mas_gain(*, h_given_m: float, beta: float, delta: float) -> float:
    """Theorem 4.1: G = H(M|m) − β Δ(m)."""
    return h_given_m - beta * delta


def evaluate_relay(config: MasIbConfig | None = None) -> RelayStats:
    cfg = config or MasIbConfig()
    h_m = cfg.relay_bits if cfg.enable_compression else cfg.full_context_bits
    h_red = context_reduction(cfg.full_context_bits, h_m) if cfg.enable_compression else 0.0
    delta = cfg.delta_loss if cfg.enable_compression else 0.0
    g = mas_gain(h_given_m=h_red, beta=cfg.beta, delta=delta)
    return RelayStats(
        context_entropy=cfg.full_context_bits,
        relay_entropy=h_m,
        context_reduction=h_red,
        info_loss=delta,
        beta=cfg.beta,
        mas_gain=g,
    )


def infinite_bandwidth_equivalent(config: MasIbConfig | None = None) -> dict:
    """Proposition 3.1: unbounded B_sys → MAS can simulate SAS (mi = Mi)."""
    cfg = config or MasIbConfig()
    no_compress = MasIbConfig(
        beta=cfg.beta,
        n_workers=cfg.n_workers,
        relay_bits=cfg.full_context_bits,
        full_context_bits=cfg.full_context_bits,
        delta_loss=0.0,
        enable_compression=False,
    )
    stats = evaluate_relay(no_compress)
    return {
        "proposition": "3.1",
        "bsys_unbounded": True,
        "y_mas_equals_y_sas": True,
        "relay_is_full_context": True,
        "mas_gain": stats.mas_gain,
        "note": "Nontrivial MAS effects require bounded relays",
    }
