"""Shared utilities for the block_streaming package."""

from __future__ import annotations

import itertools
from typing import Any

import torch
from torch import nn

# Kept local to avoid importing ``pool`` (``pool`` imports this module).
BlockLayout = dict[str, tuple[torch.Size, torch.dtype]]


def resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


def assign_tensor_to_module(root: nn.Module, dotted_name: str, tensor: torch.Tensor) -> None:
    """Assign *tensor* to the parameter/buffer at *dotted_name* inside *root*.
    Unlike ``param.data = tensor``, this works even when the existing parameter
    lives on the ``meta`` device (which has an incompatible storage type).
    """
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = tensor
    else:
        raise AttributeError(f"{leaf} is not a parameter or buffer of {type(parent).__name__}")


def build_pool_layout(block: nn.Module, dtype: torch.dtype) -> BlockLayout:
    """Derive a buffer layout from a block's parameters and buffers.
    Works on meta-device blocks (shapes are valid regardless of device).
    The *dtype* argument overrides each tensor's dtype so the pool matches
    the target inference precision.
    """
    layout: BlockLayout = {}
    for name, tensor in itertools.chain(block.named_parameters(), block.named_buffers()):
        layout[name] = (tensor.shape, dtype)
    return layout


def build_pool_layouts(blocks: nn.ModuleList, dtype: torch.dtype) -> list[BlockLayout]:
    """One layout dict per block (required when blocks are not shape-identical)."""
    return [build_pool_layout(block, dtype) for block in blocks]


def merge_block_layouts_with_checkpoint(
    meta_layouts: list[BlockLayout],
    block_tensors: dict[int, dict[str, torch.Tensor]],
    inference_dtype: torch.dtype,
) -> list[BlockLayout]:
    """Use safetensors shapes for each block where available.

    Meta-device stacks (notably Gemma 4) can expose identical parameter
    shapes on every layer while the checkpoint uses mixed widths. GPU/CPU
    streaming pools must match the checkpoint, not the meta placeholder.
    """
    out: list[BlockLayout] = []
    for i, ml in enumerate(meta_layouts):
        ck = block_tensors.get(i, {})
        merged: BlockLayout = {}
        for name in ml:
            if name in ck:
                merged[name] = (ck[name].shape, inference_dtype)
            else:
                merged[name] = ml[name]
        out.append(merged)
    return out


def layouts_homogeneous(layouts: list[BlockLayout]) -> bool:
    """Return True if every block shares the same parameter names and tensor shapes."""
    if not layouts:
        return True
    ref = layouts[0]
    for other in layouts[1:]:
        if set(ref) != set(other):
            return False
        for k in ref:
            if ref[k][0] != other[k][0]:
                return False
    return True


def layout_signature(layout: BlockLayout) -> frozenset[tuple[str, tuple[int, ...], str]]:
    """Stable fingerprint for matching GPU buffers to a block layout."""
    return frozenset(
        (name, tuple(shape), str(dt)) for name, (shape, dt) in sorted(layout.items())
    )


def allocate_buffer(layout: BlockLayout, device: torch.device, pin_memory: bool = False) -> dict[str, torch.Tensor]:
    """Allocate a single buffer dict matching *layout*."""
    return {
        name: torch.empty(shape, dtype=dtype, device=device, pin_memory=pin_memory)
        for name, (shape, dtype) in layout.items()
    }
