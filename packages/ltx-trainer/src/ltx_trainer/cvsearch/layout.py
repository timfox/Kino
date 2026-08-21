"""Scope notes for CVSearch reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No MLLM or SAM 3 weights: routing uses stub confidences and paper hyperparameters only.",
    "SLIC superpixels and agglomerative clustering are simplified; full feature-map SGAP is external.",
    "Benchmark throughput numbers are quoted from Table 5, not measured in this package.",
    "Training-free inference loop (Algorithm 1) is documented but not wired to Qwen2.5-VL runtime.",
)
