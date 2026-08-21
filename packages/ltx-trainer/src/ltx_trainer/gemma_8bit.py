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
from ltx_trainer.nvml_safe_cuda import apply_nvml_safe_cuda_patches, disable_transformers_allocator_warmup

# GPUs at or below this size use device_map=auto + partial CPU offload (e.g. RTX 4000 24 GiB).
_TIGHT_GPU_BYTES = 48 * 1024**3


def load_8bit_gemma(
    gemma_model_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    *,
    device: str | int | torch.device | None = None,
) -> GemmaTextEncoder:
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
    disable_transformers_allocator_warmup()

    gemma_path = _find_gemma_subpath(gemma_model_path, "model*.safetensors")
    tokenizer_path = _resolve_tokenizer_dir(gemma_model_path)

    weight_file = str(find_matching_file(gemma_model_path, "model*.safetensors"))
    gemma_cfg = resolve_gemma_checkpoint_config((weight_file,))
    encode_max_len = effective_gemma_encode_max_length(gemma_cfg)

    force_cpu = os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes")
    pin_map = "cpu" if force_cpu else _resolve_8bit_device_map(device)
    load_plan = _resolve_8bit_load_plan(pin_map)

    if force_cpu:
        from ltx_trainer import logger

        logger.warning(
            "GOPEX_FORCE_NVML_SAFE: loading 8-bit Gemma on CPU (caption encode is slower; avoids NVML allocator crashes)."
        )

    def _load(plan: _EightBitLoadPlan) -> Any:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=plan.cpu_offload,
        )
        kwargs: dict[str, Any] = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,
            "device_map": plan.device_map,
            "local_files_only": True,
        }
        if plan.max_memory is not None:
            kwargs["max_memory"] = plan.max_memory
        with _suppress_accelerate_memory_warnings():
            return Gemma4ForConditionalGeneration.from_pretrained(gemma_path, **kwargs)

    gemma_model: Any
    try:
        gemma_model = _load(load_plan)
    except (ValueError, torch.cuda.OutOfMemoryError) as exc:
        if _should_retry_with_cpu_offload(exc, load_plan):
            from ltx_trainer import logger

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            logger.warning(
                "8-bit Gemma load failed (%s); retrying with CPU offload and device_map=auto.",
                exc.__class__.__name__,
            )
            gemma_model = _load(_tight_gpu_offload_plan(pin_map))
        else:
            raise

    tokenizer = LTXVGemmaTokenizer(tokenizer_path, encode_max_len)

    return GemmaTextEncoder(
        tokenizer=tokenizer,
        model=gemma_model,
        dtype=dtype,
    )


class _EightBitLoadPlan:
    __slots__ = ("cpu_offload", "device_map", "max_memory")

    def __init__(
        self,
        *,
        device_map: dict[str, int] | str,
        max_memory: dict[int | str, str] | None,
        cpu_offload: bool,
    ) -> None:
        self.device_map = device_map
        self.max_memory = max_memory
        self.cpu_offload = cpu_offload


def _resolve_8bit_load_plan(pin_map: dict[str, int] | str) -> _EightBitLoadPlan:
    """Build ``from_pretrained`` placement for 8-bit Gemma."""
    if pin_map == "cpu":
        return _EightBitLoadPlan(device_map="cpu", max_memory=None, cpu_offload=True)
    cpu_offload = _want_cpu_offload(pin_map)
    if cpu_offload and _is_tight_gpu(pin_map):
        # Pinned GPU + partial CPU offload (no device_map=auto on first try — auto spills to GPU 0).
        return _EightBitLoadPlan(device_map=pin_map, max_memory=None, cpu_offload=True)
    return _EightBitLoadPlan(device_map=pin_map, max_memory=None, cpu_offload=cpu_offload)


def _tight_gpu_offload_plan(pin_map: dict[str, int] | str) -> _EightBitLoadPlan:
    """Retry plan: auto map with every non-target GPU capped at 0 GiB (train/vLLM on GPU 0)."""
    idx = _gpu_index_from_map(pin_map)
    if idx is None:
        return _EightBitLoadPlan(device_map=pin_map, max_memory=None, cpu_offload=True)
    total = _gpu_total_bytes(idx)
    gib = 16
    if total:
        gib = max(8, int(total / (1024**3)) - 4)
    return _EightBitLoadPlan(
        device_map="auto",
        max_memory=_max_memory_block_other_gpus(idx, gib),
        cpu_offload=True,
    )


def _max_memory_block_other_gpus(target_idx: int, target_gib: int) -> dict[int | str, str]:
    mem: dict[int | str, str] = {"cpu": "200GiB"}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem[i] = f"{target_gib}GiB" if i == target_idx else "0GiB"
    else:
        mem[target_idx] = f"{target_gib}GiB"
    return mem


def _want_cpu_offload(pin_map: dict[str, int] | str) -> bool:
    if os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("GOPEX_GEMMA_8BIT_CPU_OFFLOAD", "").strip().lower() in ("1", "true", "yes"):
        return True
    return _is_tight_gpu(pin_map)


def _is_tight_gpu(pin_map: dict[str, int] | str) -> bool:
    idx = _gpu_index_from_map(pin_map)
    if idx is None:
        return False
    total = _gpu_total_bytes(idx)
    return total is not None and total <= _TIGHT_GPU_BYTES


def _gpu_index_from_map(device_map: dict[str, int] | str) -> int | None:
    if isinstance(device_map, dict) and "" in device_map:
        return int(device_map[""])
    return None


def _gpu_total_bytes(index: int) -> int | None:
    if not torch.cuda.is_available():
        return None
    if index < 0 or index >= torch.cuda.device_count():
        return None
    return int(torch.cuda.get_device_properties(index).total_memory)


def _should_retry_with_cpu_offload(exc: BaseException, plan: _EightBitLoadPlan) -> bool:
    if plan.max_memory is not None:
        return False
    msg = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, ValueError) and "cpu" in msg and "disk" in msg


def _resolve_8bit_device_map(device: str | int | torch.device | None = None) -> dict[str, int] | str:
    """Pin 8-bit Gemma to one GPU.

    ``device_map="auto"`` on multi-GPU hosts often splits layers across cards; bitsandbytes then
    errors unless ``llm_int8_enable_fp32_cpu_offload`` is set. Prefer explicit ``device`` from
  preprocess (``cuda:1``), then ``GOPEX_GEMMA_CUDA_DEVICE``, then cuda:0.
    """
    if device is not None:
        dev_s = str(device)
        if dev_s.startswith("cuda:"):
            return {"": int(dev_s.split(":", 1)[1])}
        if isinstance(device, int):
            return {"": device}
        if dev_s.isdigit():
            return {"": int(dev_s)}
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
