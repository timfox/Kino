"""Entropy-guided token routing + confidence gating (Sec. 4.2, Eq. 4–8)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.entroad.config import EntroADConfig
from ltx_trainer.entroad.entropy import minmax_normalize_entropy


def patch_anomaly_pseudo_probs(
    projected_patches: np.ndarray,
    memory_keys: np.ndarray,
    memory_values: np.ndarray,
    *,
    quantile: float = 0.1,
) -> np.ndarray:
    """Eq. (4): retrieval similarities → pseudo-probability p_i."""
    r = np.asarray(projected_patches, dtype=np.float64)
    k = np.asarray(memory_keys, dtype=np.float64)
    v = np.asarray(memory_values, dtype=np.float64)
    sim = r @ k.T
    thr = np.quantile(sim, quantile)
    sim = np.where(sim < thr, -1e9, sim)
    w = np.exp(sim - sim.max(axis=1, keepdims=True))
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    resp = w @ v  # (N, 2) normal/anomaly
    if resp.shape[-1] < 2:
        return np.zeros(r.shape[0], dtype=np.float64)
    return resp[:, 1].astype(np.float64)


def routing_weights(
    p: np.ndarray,
    e_hat: np.ndarray,
    *,
    temperature: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (5): logits then spatial softmax → w_n, w_a."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    e_hat = np.asarray(e_hat, dtype=np.float64).reshape(-1)
    t = max(temperature, 1e-4)
    ra = p * (e_hat** (1.0 / t))
    rn = (1.0 - p) * ((1.0 - e_hat) ** (1.0 / t))
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        ex = np.exp(x)
        return ex / (ex.sum() + 1e-8)

    return softmax(rn), softmax(ra)


def aggregate_tokens(
    patch_features: np.ndarray,
    w_n: np.ndarray,
    w_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (6): normal token t_n and raw anomaly token t_raw."""
    z = np.asarray(patch_features, dtype=np.float64)
    w_n = np.asarray(w_n, dtype=np.float64).reshape(-1)
    w_a = np.asarray(w_a, dtype=np.float64).reshape(-1)
    t_n = (w_n[:, None] * z).sum(axis=0)
    t_raw = (w_a[:, None] * z).sum(axis=0)
    return t_n, t_raw


def confidence_gate(
    p: np.ndarray,
    e_hat: np.ndarray,
    *,
    cfg: EntroADConfig | None = None,
) -> float:
    """Eq. (7): g = sigmoid(p_max - tau) * (k0 + k1 * std(e_hat))."""
    cfg = cfg or EntroADConfig()
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    e_hat = np.asarray(e_hat, dtype=np.float64).reshape(-1)
    p_max = float(p.max()) if p.size else 0.0
    sigma_e = float(e_hat.std()) if e_hat.size else 0.0
    x = p_max - cfg.gate_tau
    sig = 1.0 / (1.0 + np.exp(-x))
    return float(sig * (cfg.gate_k0 + cfg.gate_k1 * sigma_e))


def routed_token_pair(
    patch_features: np.ndarray,
    projected_patches: np.ndarray,
    structural_entropy: np.ndarray,
    memory_keys: np.ndarray,
    memory_values: np.ndarray,
    *,
    cfg: EntroADConfig | None = None,
) -> dict[str, Any]:
    """Full Stage-2 token construction (Eq. 4–8)."""
    cfg = cfg or EntroADConfig()
    p = patch_anomaly_pseudo_probs(projected_patches, memory_keys, memory_values)
    e_hat = minmax_normalize_entropy(structural_entropy)
    n = min(len(p), len(e_hat), len(patch_features))
    p = p[:n]
    e_hat = e_hat[:n]
    patch_features = np.asarray(patch_features, dtype=np.float64)[:n]
    w_n, w_a = routing_weights(p, e_hat, temperature=cfg.router_temperature)
    t_n, t_raw = aggregate_tokens(patch_features, w_n, w_a)
    g = confidence_gate(p, e_hat, cfg=cfg)
    t_a = g * t_raw
    return {
        "t_n": t_n,
        "t_a": t_a,
        "gate": g,
        "p_max": float(p.max()) if p.size else 0.0,
        "entropy_std": float(e_hat.std()) if e_hat.size else 0.0,
    }
