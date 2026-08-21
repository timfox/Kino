"""Surgical cross-attention layer IDs (Sec. 3.4, Fig. 3)."""

from __future__ import annotations

# Arc2Face U-Net cross-attention blocks with highest identity separation
SURGICAL_BLOCKS: tuple[str, ...] = (
    "down_2",  # third downsampling block (D2)
    "mid",  # middle block (M)
    "up_1",  # first upsampling block (U1)
    "up_2",  # second upsampling block (U2)
)

FULL_CA_PARAM_FRACTION: float = 0.0511
SURGICAL_PARAM_FRACTION: float = 0.0429


def is_surgical_param(name: str) -> bool:
    """Heuristic: train only cross-attention in surgical blocks."""
    name_l = name.lower()
    if "attn" not in name_l and "attention" not in name_l:
        return False
    return any(block in name_l for block in ("down_2", "mid", "up_1", "up_2"))
