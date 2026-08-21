"""NAS-VAR architecture and training layout."""

from __future__ import annotations

SCALE_CHAIN: tuple[str, ...] = (
    "Q32 -> Q16",
    "Q16 -> Q8",
    "Q8 -> Q4",
    "Q4 -> Q2",
    "Q2 -> QFS",
)

ARCHITECTURE_NOTES: tuple[str, ...] = (
    "AQ-VAE: additive multi-input hierarchy from multiple acceleration levels (not VAR residual RQ).",
    "Shared codebook; label-conditioned encoder (R + sampling pattern via FiLM).",
    "Cross-attentive VAR transformer: encoder features at 64/32/16 injected via cross-attention.",
    "Inference from 32x undersampled input only; autoregressive next-acceleration-scale tokens.",
    "On-policy privileged distillation: teacher sees fully sampled x; reverse KL on student rollouts.",
)

LIMITATIONS: tuple[str, ...] = (
    "Evaluated at extreme R=32; may be too aggressive for routine clinical use.",
    "Discrete tokenizer bottleneck limits fidelity at lower acceleration factors.",
    "Argmax decoding required for best results; stochastic sampling underperforms.",
)

BASELINES: tuple[str, ...] = (
    "UNet",
    "SwinUNet",
    "E2EVarnet",
    "RecurrentVarnet",
    "DiffuseRecon",
    "MDPG",
    "MambaRecon",
)
