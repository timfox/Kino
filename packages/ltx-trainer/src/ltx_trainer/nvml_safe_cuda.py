"""Work around broken NVML (driver/library mismatch) for Hugging Face + PyTorch CUDA load.

When ``nvidia-smi`` fails or ``pynvml.nvmlInit()`` errors, Transformers'
``caching_allocator_warmup`` can trip PyTorch's ``CUDACachingAllocator`` NVML assert
during 8-bit Gemma load. We no-op that warmup and fall back ``mem_get_info`` to
``torch.cuda.get_device_properties``.
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

_PATCHED = False


def nvidia_smi_ok() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def nvml_healthy() -> bool:
    """True when the driver stack is consistent (``nvidia-smi`` and PyTorch CUDA alloc)."""
    if os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not nvidia_smi_ok():
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        torch.cuda.empty_cache()
        _ = torch.empty(64, device="cuda:0")
        del _
        return True
    except RuntimeError as exc:
        if "nvmlInit" in str(exc) or "NVML_SUCCESS" in str(exc):
            return False
        raise


def apply_nvml_safe_cuda_patches(*, force: bool = False) -> bool:
    """Patch Transformers/PyTorch CUDA helpers when NVML is broken. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("GOPEX_DISABLE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not force and nvml_healthy():
        return False

    _patch_transformers_caching_allocator_warmup()
    _patch_torch_cuda_mem_get_info()
    _PATCHED = True
    logger.warning(
        "NVML unavailable or mismatched — disabled Transformers caching_allocator_warmup "
        "and patched torch.cuda.mem_get_info (CUDA via PyTorch still works; reboot fixes NVML)."
    )
    return True


def _patch_transformers_caching_allocator_warmup() -> None:
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        return

    def _noop_caching_allocator_warmup(*_args, **_kwargs) -> None:
        return

    modeling_utils.caching_allocator_warmup = _noop_caching_allocator_warmup


def _patch_torch_cuda_mem_get_info() -> None:
    import torch

    if not torch.cuda.is_available():
        return

    cuda = torch.cuda
    if getattr(cuda.mem_get_info, "_gopex_nvml_safe", False):
        return
    original = cuda.mem_get_info

    def mem_get_info(device=None):
        try:
            return original(device)
        except Exception:
            idx = device if device is not None else cuda.current_device()
            if isinstance(idx, torch.device):
                idx = idx.index if idx.index is not None else cuda.current_device()
            props = cuda.get_device_properties(idx)
            total = int(props.total_memory)
            reserved = cuda.memory_reserved(idx)
            free = max(0, total - reserved)
            return free, total

    mem_get_info._gopex_nvml_safe = True  # type: ignore[attr-defined]
    cuda.mem_get_info = mem_get_info  # type: ignore[method-assign]


def embeddings_processor_device(preferred: str) -> str:
    """Where to load LTX ``EmbeddingsProcessor`` weights during ``process_captions``."""
    if os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes"):
        return "cpu"
    if not nvml_healthy():
        return "cpu"
    return preferred
