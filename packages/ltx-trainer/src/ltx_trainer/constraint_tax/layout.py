"""Harness stages and stub limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "Deterministic task generator (StructReason-Small families)",
    "Fix problem instances; vary output-control mode only",
    "Backend router: MLX/HF freeform vs vLLM/SGLang constrained decoding",
    "Parse + schema validate + answer normalize + executable/trace check",
    "Record validity, accuracy, wrong-valid-schema, latency, tokens",
    "Compute constraint tax vs prompt_json (or mode-specific baseline)",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — no vLLM/SGLang/MLX live runs or 15k-generation replay.",
    "Table excerpts and tax math are implemented; toy smoke uses synthetic paired rows.",
    "Does not ship StructReason-Small JSONL artifacts from the paper experiments.",
)
