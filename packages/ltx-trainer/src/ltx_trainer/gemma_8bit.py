# ruff: noqa: PLC0415

"""
8-bit Gemma text encoder loading utilities.
This module provides functionality for loading the Gemma text encoder in 8-bit precision
using bitsandbytes, which significantly reduces GPU memory usage.
Example usage:
    from ltx_trainer.gemma_8bit import load_8bit_gemma
    text_encoder = load_8bit_gemma(gemma_model_path="/path/to/gemma")
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ltx_core.text_encoders.gemma.config import (
    effective_gemma_encode_max_length,
    resolve_gemma_checkpoint_config,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file
from ltx_trainer.nvml_safe_cuda import apply_nvml_safe_cuda_patches


def load_8bit_gemma(gemma_model_path: str | Path, dtype: torch.dtype = torch.bfloat16) -> GemmaTextEncoder:
    """Load the Gemma text encoder in 8-bit precision using bitsandbytes.
    Only the Gemma LLM backbone is loaded here.  The embeddings processor
    (feature extractor + connectors) should be loaded separately via
    :func:`ltx_trainer.model_loader.load_embeddings_processor`.
    Args:
        gemma_model_path: Path to Gemma model directory
        dtype: Data type for non-quantized model weights
    Returns:
        GemmaTextEncoder with 8-bit quantized Gemma backbone
    Raises:
        ImportError: If bitsandbytes is not installed
        FileNotFoundError: If required model files are not found
    """
    try:
        from transformers import BitsAndBytesConfig, Gemma4ForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires bitsandbytes. Install it with: uv pip install bitsandbytes"
        ) from e

    apply_nvml_safe_cuda_patches()

    gemma_path = _find_gemma_subpath(gemma_model_path, "model*.safetensors")
    tokenizer_path = _resolve_tokenizer_dir(gemma_model_path)

    weight_file = str(find_matching_file(gemma_model_path, "model*.safetensors"))
    gemma_cfg = resolve_gemma_checkpoint_config((weight_file,))
    encode_max_len = effective_gemma_encode_max_length(gemma_cfg)

    force_cpu = os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes")
    device_map = "cpu" if force_cpu else _resolve_8bit_device_map()
    allow_cpu_offload = force_cpu or os.environ.get("GOPEX_GEMMA_8BIT_CPU_OFFLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_cpu:
        from ltx_trainer import logger

        logger.warning(
            "GOPEX_FORCE_NVML_SAFE: loading 8-bit Gemma on CPU (caption encode is slower; avoids NVML allocator crashes)."
        )

    def _load(*, cpu_offload: bool) -> Any:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
        with _suppress_accelerate_memory_warnings():
            return Gemma4ForConditionalGeneration.from_pretrained(
                gemma_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                local_files_only=True,
            )

    try:
        gemma_model = _load(cpu_offload=allow_cpu_offload)
    except ValueError as exc:
        msg = str(exc).lower()
        if not allow_cpu_offload and "cpu" in msg and "disk" in msg:
            gemma_model = _load(cpu_offload=True)
        else:
            raise

    tokenizer = LTXVGemmaTokenizer(tokenizer_path, encode_max_len)

    return GemmaTextEncoder(
        tokenizer=tokenizer,
        model=gemma_model,
        dtype=dtype,
    )


def _resolve_8bit_device_map() -> dict[str, int] | str:
    """Pin 8-bit Gemma to one GPU.

    ``device_map="auto"`` on multi-GPU hosts often splits layers across cards; bitsandbytes then
    errors unless ``llm_int8_enable_fp32_cpu_offload`` is set. Default: all weights on cuda:0.
    Override with ``GOPEX_GEMMA_CUDA_DEVICE`` (index) or ``GOPEX_GEMMA_DEVICE_MAP`` (e.g. ``0``).
    """
    raw = os.environ.get("GOPEX_GEMMA_DEVICE_MAP", "").strip()
    if raw:
        if raw.isdigit():
            return {"": int(raw)}
        return raw
    dev = os.environ.get("GOPEX_GEMMA_CUDA_DEVICE", "0").strip() or "0"
    try:
        return {"": int(dev)}
    except ValueError:
        return "auto"


def _find_gemma_subpath(root_path: str | Path, pattern: str) -> str:
    """Find a file matching a glob pattern and return its parent directory."""
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return str(matches[0].parent)


def _resolve_tokenizer_dir(root_path: str | Path) -> str:
    """Directory passed to :class:`~ltx_core.text_encoders.gemma.tokenizer.LTXVGemmaTokenizer`.

    Gemma 4 Hugging Face trees ship ``tokenizer.json`` (not legacy ``tokenizer.model``).
    """
    root = Path(root_path).expanduser().resolve()
    if (root / "tokenizer.json").is_file() or (root / "tokenizer_config.json").is_file():
        return str(root)
    for pattern in ("tokenizer.json", "tokenizer.model"):
        matches = list(root.rglob(pattern))
        if matches:
            return str(matches[0].parent)
    raise FileNotFoundError(
        f"No tokenizer.json or tokenizer.model found under {root_path}. "
        "Point text_encoder_path at a full Gemma 4 HF snapshot directory."
    )


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    """Temporarily suppress INFO warnings from accelerate about memory allocation."""
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)
