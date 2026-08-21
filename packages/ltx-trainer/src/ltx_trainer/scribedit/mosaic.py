"""Multi-task mosaicking for higher-order tuple supervision (Sec. 4.2, Fig. 2c)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MosaicSample:
    """Composite ⟨multi-scribble image, multi-instruction, multi-edit target⟩."""

    scribbled_input: Tensor
    target: Tensor
    instructions: list[str]
    layout: str
    source_colors: list[str]


def _paste_grid(tiles: list[Tensor], layout: str) -> Tensor:
    """Concatenate (C,H,W) tiles into one image."""
    if layout == "1x2":
        return torch.cat(tiles, dim=2)
    if layout == "2x1":
        return torch.cat(tiles, dim=1)
    if layout == "2x2":
        top = torch.cat(tiles[:2], dim=2)
        bottom = torch.cat(tiles[2:], dim=2)
        return torch.cat([top, bottom], dim=1)
    raise ValueError(f"unknown layout: {layout}")


def multi_task_mosaic(
    inputs: list[Tensor],
    targets: list[Tensor],
    instructions: list[str],
    *,
    layout: str = "1x2",
    colors: list[str] | None = None,
    shuffle_instructions: bool = True,
) -> MosaicSample:
    """Mosaic k single-task samples; instructions concatenated in random order."""
    k = len(inputs)
    expected = {"1x2": 2, "2x1": 2, "2x2": 4}.get(layout)
    if expected is None or k != expected:
        raise ValueError(f"layout {layout} requires {expected} samples, got {k}")
    if len(targets) != k or len(instructions) != k:
        raise ValueError("inputs, targets, and instructions must have equal length")

    palette = colors or [f"color_{i}" for i in range(k)]
    if len(set(palette)) != k:
        raise ValueError("scribble colors must be mutually distinct (Sec. 4.2)")

    order = list(range(k))
    if shuffle_instructions:
        random.shuffle(order)
    ordered_instr = [instructions[i] for i in order]

    return MosaicSample(
        scribbled_input=_paste_grid(inputs, layout),
        target=_paste_grid(targets, layout),
        instructions=ordered_instr,
        layout=layout,
        source_colors=palette,
    )


def format_multi_instruction(instructions: list[str]) -> str:
    """Join k task prompts into one multi-task text conditioning string."""
    return " ".join(instr.strip() for instr in instructions if instr.strip())
