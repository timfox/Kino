"""Environment-aware routing for plug-and-play inference (Sec. 4.3)."""

from __future__ import annotations

from typing import Any


def route_decision(dirty_probability: float, *, threshold: float = 0.5) -> str:
    """Route to Mega-ASR LoRA branch when p_dirty >= γ (default γ=0.5)."""
    return "mega_asr_lora" if dirty_probability >= threshold else "qwen3_asr_base"


def router_architecture_summary() -> dict[str, Any]:
    """Table 18 / Sec. D.2 — lightweight MFCC binary classifier."""
    return {
        "input": "80-dim log-Mel spectrogram",
        "sample_rate_hz": 16000,
        "max_duration_s": 30,
        "frontend": "1D conv + 2x temporal downsample",
        "encoder": "1-layer Transformer (4 heads, d=128)",
        "pooling": "attention pooling",
        "output": "clean (0) / degraded (1)",
        "dev_accuracy_pct": 99.5,
    }
