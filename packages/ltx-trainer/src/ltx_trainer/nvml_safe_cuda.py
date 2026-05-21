"""Work around broken NVML (driver/library mismatch) for Hugging Face + PyTorch CUDA load.

When ``nvidia-smi`` fails or ``pynvml.nvmlInit()`` errors, Transformers'
``caching_allocator_warmup`` and PyTorch's default ``CUDACachingAllocator`` can trip
NVML asserts (including during ``Adam.step()`` state allocation).

Mitigations applied when NVML is unhealthy:

1. ``PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`` (must be set before first CUDA use)
2. No-op Transformers ``caching_allocator_warmup``
3. Fallback ``torch.cuda.mem_get_info`` via device properties

Reboot / fix driver-library mismatch is still the proper long-term fix.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_PATCHED = False
_CUDA_MALLOC_ASYNC_CONF = "backend:cudaMallocAsync"


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


def _nvml_safe_forced() -> bool:
    return os.environ.get("GOPEX_FORCE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes")


def _set_cuda_malloc_async_allocator() -> None:
    """Use cudaMallocAsync so CUDACachingAllocator does not require NVML on alloc."""
    cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if "backend:cudaMallocAsync" in cur:
        return
    merged = f"{cur},{_CUDA_MALLOC_ASYNC_CONF}" if cur else _CUDA_MALLOC_ASYNC_CONF
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = merged
    logger.info("Set PYTORCH_CUDA_ALLOC_CONF=%s (NVML-safe training alloc)", merged)


def nvml_healthy() -> bool:
    """True when the driver stack is consistent (``nvidia-smi`` and PyTorch CUDA alloc)."""
    if _nvml_safe_forced():
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
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device="cuda:0"))], lr=1e-3)
        opt.step()
        return True
    except RuntimeError as exc:
        if "nvmlInit" in str(exc) or "NVML_SUCCESS" in str(exc):
            return False
        raise


def apply_nvml_safe_cuda_patches(*, force: bool = False) -> bool:
    """Patch allocators/helpers when NVML is broken. Idempotent. Call before CUDA work."""
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("GOPEX_DISABLE_NVML_SAFE", "").strip().lower() in ("1", "true", "yes"):
        return False
    if not force and not _nvml_safe_forced() and nvml_healthy():
        return False

    _set_cuda_malloc_async_allocator()
    _patch_transformers_caching_allocator_warmup()
    _patch_torch_cuda_mem_get_info()
    _PATCHED = True
    logger.warning(
        "NVML unavailable or mismatched — using cudaMallocAsync allocator, disabled "
        "Transformers caching_allocator_warmup, patched mem_get_info (reboot fixes NVML)."
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
    """Where to load LTX ``EmbeddingsProcessor`` during caption encode / validation cache."""
    if _nvml_safe_forced() or not nvidia_smi_ok():
        return "cpu"
    return preferred


def move_gemma_encode_outputs_to_device(
    hidden_states: tuple | torch.Tensor | Sequence,
    attention_mask: torch.Tensor,
    device: str | torch.device,
) -> tuple[tuple | torch.Tensor, torch.Tensor]:
    """Align Gemma ``encode()`` outputs with the embeddings processor device."""
    import torch

    dev = torch.device(device)
    if isinstance(hidden_states, torch.Tensor):
        hs: tuple[torch.Tensor, ...] | torch.Tensor = hidden_states.to(dev)
    else:
        hs = tuple(t.to(dev) for t in hidden_states)
    return hs, attention_mask.to(dev)
