"""Scope notes for SegCompass reference stub."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No MLLM/SAE/ViT-H training: SAE sparsity, GRPO, and benchmark tables only.",
    "VERL/FSDP/vLLM distributed training is external to this package.",
    "cIoU/gIoU metrics are quoted from the paper, not reproduced locally.",
    "Frozen SAE from OBELICS pretrain is assumed, not implemented here.",
)
