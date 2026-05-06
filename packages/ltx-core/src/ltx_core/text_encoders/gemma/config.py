"""Gemma 4 checkpoint configuration helpers for the LTX text encoder.

Architecture and weights must come from the same Hugging Face Gemma 4 release you use locally.
The LTX diffusion stack (feature extractor ``flat_dim``, connectors) must be trained or exported
for that encoder width and layer count; swapping Gemma 3 for Gemma 4 without a matching LTX checkpoint
will not work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader


def resolve_gemma_checkpoint_config(weight_paths: tuple[str, ...]) -> dict[str, Any]:
    """Return the Hugging Face ``config.json`` payload used to build ``Gemma4ForConditionalGeneration``.

    Resolution order:

    1. ``config`` metadata on the first ``.safetensors`` shard (HF exports).
    2. ``config.json`` next to that shard (same directory as the weight files).

    Raises:
        ValueError: If no configuration is found or ``model_type`` is not ``gemma4``.
    """
    if not weight_paths:
        raise ValueError("Gemma weight_paths must be non-empty")
    first = weight_paths[0]
    loader = SafetensorsModelStateDictLoader()
    cfg: dict[str, Any] = loader.metadata(first) or {}
    if cfg.get("model_type") != "gemma4":
        alt = Path(first).parent / "config.json"
        if alt.is_file():
            cfg = json.loads(alt.read_text(encoding="utf-8"))
    if cfg.get("model_type") != "gemma4":
        msg = (
            "Could not load a Gemma 4 config: expected model_type 'gemma4' from safetensors metadata "
            f"or {Path(first).parent / 'config.json'}. Place Hugging Face Gemma 4 config.json beside the weights, "
            "or use safetensors shards that include HF config metadata."
        )
        raise ValueError(msg)
    return cfg
