"""OpenAI-compatible StepFun API helpers (no network in unit tests)."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.step37_flash.constants import API_MODEL_ID, API_REGIONS

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant provided by StepFun. You are good at Chinese, English, "
    "and many other languages, and you can see, think, and act to help users get things done."
)


def resolve_region(region: str | None = None) -> str:
    r = (region or os.environ.get("STEP_REGION", "global")).lower()
    if r not in API_REGIONS:
        raise ValueError(f"region must be one of {tuple(API_REGIONS)}, got {r!r}")
    return r


def api_env_card(region: str | None = None) -> dict[str, Any]:
    """Environment variables for OpenAI SDK client."""
    r = resolve_region(region)
    meta = API_REGIONS[r]
    return {
        "region": r,
        "platform_url": meta["platform"],
        "base_url": meta["base_url"],
        "model": API_MODEL_ID,
        "env": {
            "STEP_API_KEY": "sk-...",
            "STEP_BASE_URL": meta["base_url"],
            "STEP_REGION": r,
        },
        "note": "API key must match platform region or requests are unauthorized.",
    }


def build_chat_request(
    user_text: str,
    *,
    system: str = DEFAULT_SYSTEM_PROMPT,
    image_url: str | None = None,
    reasoning_level: str | None = None,
) -> dict[str, Any]:
    """OpenAI chat.completions.create payload (serializable)."""
    if image_url:
        user_content: list[dict[str, Any]] | str = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = user_text

    body: dict[str, Any] = {
        "model": API_MODEL_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    if reasoning_level:
        body["extra_body"] = {"reasoning": {"effort": reasoning_level}}
    return body


def openai_client_kwargs(region: str | None = None) -> dict[str, str]:
    r = resolve_region(region)
    return {
        "api_key": os.environ.get("STEP_API_KEY", "${STEP_API_KEY}"),
        "base_url": os.environ.get("STEP_BASE_URL", API_REGIONS[r]["base_url"]),
    }
