"""Context manager for temporary GPU model lifetime.

Yields a loaded module for inference, then moves parameters to the meta device
and clears CUDA caches so peak VRAM drops between pipeline blocks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ltx_pipelines.utils.helpers import cleanup_memory


@contextmanager
def gpu_model(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    try:
        yield model
    finally:
        model.to("meta")
        cleanup_memory()
