"""IRAF reliability gate and adaptive fusion (§2.2, Eq. 2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.iraf.config import IrafConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def reliability_gate(
    logits: np.ndarray,
    *,
    gate_scale: float = 2.0,
) -> np.ndarray:
    """g_t = 2 × Sigmoid(logit) ∈ [0, 2] (Eq. 2)."""
    return gate_scale * _sigmoid(np.asarray(logits, dtype=np.float64))


def adaptive_fusion(
    user_embed: np.ndarray,
    agent_text_embed: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    """Fused LLM input: g_t * X_t + Y_txt_t (§2.2)."""
    g = np.asarray(gate, dtype=np.float64)
    if g.ndim == 1:
        g = g[:, None]
    return g * np.asarray(user_embed, dtype=np.float64) + np.asarray(agent_text_embed, dtype=np.float64)


def gate_logits_stub(
    speaker_embed: np.ndarray,
    user_frames: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Toy causal gate predictor f(s, X_≤t | ψ) for CPU demo."""
    rng = np.random.default_rng(seed)
    t, d = user_frames.shape
    s = np.asarray(speaker_embed, dtype=np.float64).reshape(-1)
    w_s = rng.normal(scale=0.05, size=s.shape[0])
    w_x = rng.normal(scale=0.05, size=d)
    logits = np.empty(t, dtype=np.float64)
    for t_idx in range(t):
        ctx = user_frames[: t_idx + 1].mean(axis=0)
        logits[t_idx] = float(s @ w_s + ctx @ w_x)
    return logits


def gate_binary_loss(
    gate: np.ndarray,
    target_active: np.ndarray,
) -> float:
    """Auxiliary frame-level BCE on reliability targets (§2.2, weight 0.1)."""
    g = np.clip(np.asarray(gate, dtype=np.float64) / 2.0, 1e-6, 1.0 - 1e-6)
    y = np.asarray(target_active, dtype=np.float64)
    return float(-np.mean(y * np.log(g) + (1.0 - y) * np.log(1.0 - g)))


def duplex_loss_stub(
    *,
    text_ce: float = 1.2,
    audio_ce: float = 0.8,
    gate_bce: float = 0.35,
    cfg: IrafConfig | None = None,
) -> dict[str, float]:
    """Toy multi-task loss matching Eq. (1) + gate auxiliary."""
    c = cfg or IrafConfig()
    main = c.lambda_text * text_ce + c.lambda_audio * audio_ce
    total = main + c.lambda_gate * gate_bce
    return {
        "text_ce": text_ce,
        "audio_ce": audio_ce,
        "gate_bce": gate_bce,
        "main": main,
        "total": total,
        "lambda_text": c.lambda_text,
        "lambda_audio": c.lambda_audio,
        "lambda_gate": c.lambda_gate,
    }


def gate_demo(seed: int = 0, cfg: IrafConfig | None = None) -> dict[str, Any]:
    """Simulate gating under interference-dominated vs target-active frames."""
    c = cfg or IrafConfig()
    rng = np.random.default_rng(seed)
    t, d, n_s = 40, 32, 64
    speaker = rng.normal(size=n_s)
    user = rng.normal(size=(t, d))
    target_active = (rng.random(t) > 0.45).astype(np.float64)
    # Interference frames: low correlation with speaker direction
    interference = target_active < 0.5
    user[interference] += rng.normal(scale=2.0, size=(interference.sum(), d))

    logits = gate_logits_stub(speaker, user, seed=seed)
    # Bias logits toward target-active frames (supervision signal from §2.2).
    logits = logits + (target_active - 0.5) * 4.0
    gate = reliability_gate(logits, gate_scale=c.gate_scale)
    fused = adaptive_fusion(user, rng.normal(scale=0.1, size=(t, d)), gate)
    bce = gate_binary_loss(gate, target_active)

    active_gate = float(gate[target_active > 0.5].mean()) if (target_active > 0.5).any() else 0.0
    idle_gate = float(gate[target_active < 0.5].mean()) if (target_active < 0.5).any() else 0.0

    return {
        "frames": t,
        "gate_scale": c.gate_scale,
        "gate_min": float(gate.min()),
        "gate_max": float(gate.max()),
        "mean_gate_target_active": active_gate,
        "mean_gate_interference": idle_gate,
        "suppresses_interference": active_gate > idle_gate,
        "gate_bce": bce,
        "fused_shape": list(fused.shape),
        "loss": duplex_loss_stub(gate_bce=bce, cfg=c),
    }
