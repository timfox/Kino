"""GOPEX vLLM / local Gemma-Qwen stack integration plan for LLMCodec weights."""

from __future__ import annotations

from typing import Any

from ltx_trainer.llmcodec.config import LlmCodecConfig


def vllm_integration_plan(cfg: LlmCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or LlmCodecConfig()
    return {
        "goal": "Shrink vLLM-served checkpoint storage with codec-compressed safetensors sidecars",
        "phases": [
            {
                "id": "export_layers",
                "action": "Export linear weight matrices from Gemma-4 / Qwen checkpoints",
                "tool": "llmcodec compress_weight_matrix per shard",
            },
            {
                "id": "codec_pack",
                "action": "Affine + RTN + YUV420 + VVenC All-Intra at target QP",
                "artifact": "*.llmcodec.vvc sidecar per layer",
                "env": "GOPEX_LLMCODEC_QP=12",
            },
            {
                "id": "serve_decompress",
                "action": "Lazy VVdeC decode on first load; cache FP16/BF16 in RAM",
                "tool": "./scripts/vllm-serve-gopex-stack.sh (future hook)",
            },
        ],
        "targets": [
            "Gemma-4-31B text encoder weights (connector-only fine-tunes)",
            "Qwen-32B agent tier",
            "Whisper/TARQ adapters (optional)",
        ],
        "paper_models": list(c.models),
        "recommended_codec": c.default_codec.value,
        "recommended_profile": c.default_profile.value,
    }


def gopex_env_snippet() -> str:
    return "\n".join(
        [
            "export GOPEX_LLMCODEC=1",
            "export GOPEX_LLMCODEC_QP=12",
            "export GOPEX_LLMCODEC_PROFILE=All-Intra",
            "export GOPEX_LLMCODEC_CODEC=VVC/H.266",
        ]
    )
