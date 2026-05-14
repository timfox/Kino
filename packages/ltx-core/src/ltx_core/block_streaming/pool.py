"""Weight buffer pool for block streaming."""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from ltx_core.block_streaming.utils import allocate_buffer, layout_signature

# Type alias for the buffer layout used by slot allocation.
BlockLayout = dict[str, tuple[torch.Size, torch.dtype]]


class WeightPool:
    """Fixed pool of weight buffers with event-based reuse safety.
    **Homogeneous** mode: all blocks share one *layout*; *capacity* identical
    buffers are pre-allocated at construction.
    **Heterogeneous** mode: pass *per_block_layouts* (e.g. Gemma 4 mixed layer
    widths). Buffers are acquired with *block_idx* so tensor shapes match the
    active layer.
    Args:
        capacity: Max GPU/CPU buffer slots in circulation.
        device: Device for allocation.
        reuse_barrier: Called with the pending event before a buffer is reused.
        pin_memory: Pin buffers (for async H2D copies from CPU).
        layout: Single layout (mutually exclusive with *per_block_layouts*).
        per_block_layouts: Per-block layouts for heterogeneous stacks.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        reuse_barrier: Callable[[torch.cuda.Event], None],
        pin_memory: bool = False,
        *,
        layout: BlockLayout | None = None,
        per_block_layouts: list[BlockLayout] | None = None,
    ) -> None:
        self._capacity = capacity
        self._device = device
        self._pin_memory = pin_memory
        self._reuse_barrier = reuse_barrier
        self._events: dict[int, torch.cuda.Event] = {}
        self._per_block_layouts = per_block_layouts
        self._heterogeneous = per_block_layouts is not None

        if (per_block_layouts is not None) == (layout is not None):
            raise ValueError("Specify exactly one of layout= (homogeneous) or per_block_layouts= (heterogeneous)")

        if self._heterogeneous:
            if not per_block_layouts:
                raise ValueError("per_block_layouts must be non-empty")
            self._homogeneous_layout: BlockLayout | None = None
            self._free: deque[tuple[frozenset, dict[str, torch.Tensor]]] = deque()
            self._allocated = 0
        else:
            assert layout is not None
            self._homogeneous_layout = layout
            self._free: deque[dict[str, torch.Tensor]] = deque()
            for _ in range(capacity):
                self._free.append(allocate_buffer(layout, device, pin_memory))
            self._allocated = capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    def acquire(self, block_idx: int | None = None) -> dict[str, torch.Tensor]:
        """Take a free buffer, waiting any pending event before returning."""
        if self._heterogeneous:
            if block_idx is None:
                raise ValueError("block_idx is required for heterogeneous WeightPool")
            return self._acquire_heterogeneous(block_idx)
        return self._acquire_homogeneous()

    def _acquire_homogeneous(self) -> dict[str, torch.Tensor]:
        weights = self._free.popleft()
        event = self._events.pop(id(weights), None)
        if event is not None:
            self._reuse_barrier(event)
        return weights

    def _acquire_heterogeneous(self, block_idx: int) -> dict[str, torch.Tensor]:
        layout = self._per_block_layouts[block_idx]
        target_sig = layout_signature(layout)

        tmp: list[tuple[frozenset, dict[str, torch.Tensor]]] = []
        while self._free:
            sig, buf = self._free.popleft()
            if sig == target_sig:
                for item in tmp:
                    self._free.append(item)
                event = self._events.pop(id(buf), None)
                if event is not None:
                    self._reuse_barrier(event)
                return buf
            tmp.append((sig, buf))
        for item in tmp:
            self._free.append(item)

        if self._free:
            _, buf = self._free.popleft()
            event = self._events.pop(id(buf), None)
            if event is not None:
                self._reuse_barrier(event)
            return allocate_buffer(layout, self._device, self._pin_memory)

        if self._allocated < self._capacity:
            self._allocated += 1
            return allocate_buffer(layout, self._device, self._pin_memory)

        raise RuntimeError(
            "WeightPool (heterogeneous): no free buffer and at capacity — "
            "this should not happen if cache size matches pool capacity."
        )

    def release(
        self,
        weights: dict[str, torch.Tensor],
        event: torch.cuda.Event | None = None,
        *,
        block_idx: int | None = None,
    ) -> None:
        """Return a buffer to the free list.
        If *event* is given it is waited on the next :meth:`acquire`
        of this buffer, ensuring the prior operation has completed.
        """
        if event is not None:
            self._events[id(weights)] = event
        if self._heterogeneous:
            if block_idx is None:
                raise ValueError("block_idx is required for heterogeneous WeightPool")
            sig = layout_signature(self._per_block_layouts[block_idx])
            self._free.append((sig, weights))
        else:
            self._free.append(weights)
