"""GOPEX / LTX integration — fast VLM captions via Step 3.7 Flash API."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.api import api_env_card, build_chat_request
from ltx_trainer.step37_flash.constants import API_MODEL_ID


def ltx_caption_plan() -> dict[str, Any]:
    """
    Wire Step 3.7 Flash as optional backend for dataset_split_and_caption / parallel prep.

    Does not download 198B weights locally unless operator opts into vLLM serve.
    """
    return {
        "hook": "Optional fast VLM caption backend for dataset_split_and_caption",
        "env": {
            "GOPEX_CAPTION_BACKEND": "step37_flash",
            "STEP_API_KEY": "required when using API",
            "STEP_BASE_URL": "https://api.stepfun.ai/v1 or https://api.stepfun.com/v1",
            "GOPEX_STEP37_REASONING": "low|medium|high (default medium)",
        },
        "model": API_MODEL_ID,
        "api": api_env_card(),
        "sample_request": build_chat_request(
            "Describe this clip for LTX training: subject, motion, lighting, camera.",
            image_url="file://$CLIP_FIRST_FRAME",
            reasoning_level="low",
        ),
        "local_alternative": {
            "vllm": "8×GPU TP + expert parallel; see step37_flash_deploy_cmd",
            "gguf": "128GB unified memory Mac Studio / DGX Station",
        },
        "when_to_use": [
            "High-throughput caption sweeps without loading Gemma 31B on GPU 1",
            "Multimodal shot understanding (UI, charts, dense frames)",
            "256k context for long teleplay manifests",
        ],
    }
