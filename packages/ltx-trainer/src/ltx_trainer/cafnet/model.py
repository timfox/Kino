"""CAFNet and MFAAN architecture stubs (EnhancedPath, CrossAttnFusion, TemporalHead)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.cafnet.config import CafNetConfig


def depthwise_separable_block(x: np.ndarray, in_ch: int, out_ch: int) -> np.ndarray:
    """Toy 1D DSConv: (channels × time) → (out_ch × time/2)."""
    c, t = x.shape
    # depthwise: per-channel smoothing
    kernel = np.ones(3) / 3.0
    dw = np.stack([np.convolve(x[i], kernel, mode="same") for i in range(c)], axis=0)
    # pointwise: linear mix to out_ch
    w = np.random.default_rng(in_ch + out_ch).standard_normal((out_ch, c)) * 0.05
    pw = w @ dw
    pooled = pw[:, ::2][:, : max(1, t // 2)]
    return np.maximum(pooled, 0.0)


def enhanced_path(feat: np.ndarray, *, in_dim: int, rng: np.random.Generator) -> np.ndarray:
    """EnhancedPath: DSConv blocks + lightweight self-attention residual."""
    x = feat
    x = depthwise_separable_block(x, in_dim, 64)
    x = depthwise_separable_block(x, 64, 128)
    # self-attention proxy (query/key dim 16)
    q = rng.standard_normal((16, x.shape[1])) * 0.1
    k = rng.standard_normal((16, x.shape[1])) * 0.1
    attn = np.tanh(q.T @ k)
    attn = attn / (np.linalg.norm(attn) + 1e-10)
    residual = x @ attn
    alpha = 0.0  # learnable scalar init 0
    return x + alpha * residual


def cross_attention_fusion(
    mfcc_seq: np.ndarray,
    lfcc_seq: np.ndarray,
    chroma_seq: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """CrossAttnFusion: MFCC query, LFCC+Chroma key/value, gating → 128-d."""
    q = mfcc_seq.mean(axis=1)
    k = 0.5 * (lfcc_seq.mean(axis=1) + chroma_seq.mean(axis=1))
    score = float(np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-10))
    gate = 1.0 / (1.0 + np.exp(-score))
    pooled = np.stack([mfcc_seq.mean(), lfcc_seq.mean(), chroma_seq.mean()])
    gates = rng.random(3)
    gates = gates / (gates.sum() + 1e-10)
    fused = gate * (pooled @ gates) + (1 - gate) * pooled.mean()
    proj = rng.standard_normal(128) * fused
    return proj


def temporal_head(
    path_outputs: list[np.ndarray],
    *,
    rng: np.random.Generator,
    units: int = 64,
) -> tuple[float, float]:
    """BiLSTM boundary regression → normalised start/end in [0, 1]."""
    concat = np.concatenate([p.mean(axis=1) for p in path_outputs])
    if concat.size < units:
        concat = np.pad(concat, (0, units - concat.size))
    else:
        concat = concat[:units]
    w_fwd = rng.standard_normal((units, units)) * 0.05
    w_bwd = rng.standard_normal((units, units)) * 0.05
    fwd = np.tanh(w_fwd @ concat)
    bwd = np.tanh(w_bwd @ concat)
    h = np.concatenate([fwd, bwd])
    w_out = rng.standard_normal((2, h.size)) * 0.1
    raw = w_out @ h
    bounds = 1.0 / (1.0 + np.exp(-raw))
    return float(np.clip(bounds[0], 0.0, 1.0)), float(np.clip(bounds[1], 0.0, 1.0))


def cafnet_forward(
    features: dict[str, np.ndarray],
    *,
    cfg: CafNetConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Single forward pass: 3-class logits + boundary prediction."""
    cfg = cfg or CafNetConfig()
    rng = np.random.default_rng(seed)
    mfcc = enhanced_path(features["mfcc"], in_dim=cfg.mfcc_coeffs, rng=rng)
    lfcc = enhanced_path(features["lfcc"], in_dim=cfg.lfcc_coeffs, rng=rng)
    chroma = enhanced_path(features["chroma"], in_dim=cfg.chroma_bins, rng=rng)
    fused = cross_attention_fusion(mfcc, lfcc, chroma, rng=rng)
    logits_main = rng.standard_normal(3) + fused[:3]
    logits_aux = rng.standard_normal(3) + fused[3:6] if fused.size >= 6 else logits_main * 0.9
    start, end = temporal_head([features["mfcc"], features["lfcc"], features["chroma"]], rng=rng, units=cfg.bilstm_units)
    probs = np.exp(logits_main - logits_main.max())
    probs = probs / probs.sum()
    return {
        "logits_main": logits_main.tolist(),
        "logits_aux": np.asarray(logits_aux).tolist(),
        "class_probs": probs.tolist(),
        "pred_class": int(np.argmax(probs)),
        "boundary_norm": {"start": start, "end": end},
        "boundary_seconds": {"start": start * cfg.clip_seconds, "end": end * cfg.clip_seconds},
        "params": cfg.cafnet_params,
    }


def mfaan_forward(features: dict[str, np.ndarray], *, seed: int = 42) -> dict[str, Any]:
    """MFAAN binary baseline: 2D-CNN path proxy → 2-class logits."""
    cfg = CafNetConfig()
    rng = np.random.default_rng(seed)
    pools = []
    for key in ("mfcc", "lfcc", "chroma"):
        x = features[key]
        # 2D conv + adaptive pool proxy
        pools.append(x.mean(axis=1)[:128])
    concat = np.concatenate(pools)
    logits = rng.standard_normal(2) + concat[:2]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return {
        "logits": logits.tolist(),
        "class_probs": probs.tolist(),
        "pred_fake": int(np.argmax(probs)),
        "params": cfg.mfaan_params,
    }
