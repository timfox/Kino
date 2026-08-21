"""ChildVox encoder/LALM fine-tuning smoke (LoRA + weighted layer pool)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.childvox.config import ChildVoxConfig


def weighted_encoder_pool(
    layer_hiddens: np.ndarray,
    *,
    num_classes: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Figure 3: learnable layer weights → 1D conv → temporal avg → classifier."""
    rng = np.random.default_rng(seed)
    # layer_hiddens: (num_layers, time, dim)
    weights = rng.random(layer_hiddens.shape[0])
    weights = weights / weights.sum()
    pooled = np.tensordot(weights, layer_hiddens, axes=(0, 0))  # time × dim
    conv = np.tanh(rng.standard_normal(pooled.shape[1]) @ pooled.mean(axis=0) * 0.05)
    logits = rng.standard_normal(num_classes) + conv[:num_classes] if conv.size >= num_classes else rng.standard_normal(num_classes)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return {"logits": logits.tolist(), "probs": probs.tolist(), "pred": int(np.argmax(probs))}


def lora_smoke(num_params_base: int, *, rank: int = 64) -> dict[str, int]:
    """Approximate LoRA adapter parameter count for rank-64 FFN injection."""
    # toy: 2 * rank * hidden per injected layer
    hidden = 768
    layers = 12
    lora_params = 2 * rank * hidden * layers
    return {"base_params": num_params_base, "lora_params": lora_params, "rank": rank}


def model_catalog() -> list[dict[str, Any]]:
    """Table 2 — pre-trained models in ChildVox."""
    return [
        {"name": "SSAST-Base", "objective": "self-supervised", "params": "89M", "family": "audio"},
        {"name": "voc2vec-HuBERT", "objective": "self-supervised", "params": "89M", "family": "vocalization"},
        {"name": "WavLM-Large", "objective": "self-supervised", "params": "316M", "family": "speech"},
        {"name": "Whisper-Base", "objective": "ASR/VAD/LID", "params": "20.1M", "family": "asr"},
        {"name": "Whisper-Small", "objective": "ASR/VAD/LID", "params": "88M", "family": "asr"},
        {"name": "Whisper-Large-v3", "objective": "ASR/VAD/LID", "params": "635M", "family": "asr"},
        {"name": "Qwen2-Audio-Instruct", "objective": "multi-modal LALM", "params": "7B", "family": "lalm"},
        {"name": "AudioFlamingo3", "objective": "multi-modal LALM", "params": "8B", "family": "lalm"},
    ]
