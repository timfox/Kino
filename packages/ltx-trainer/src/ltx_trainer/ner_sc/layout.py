"""Scope notes for NeR-SC reference implementation."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "No PyTorch training loop: forward modules are NumPy reference stubs for agents and smoke tests.",
    "DSCVC/VCD frame IO and H.264/x265 RD curves are table excerpts from the paper, not re-trained baselines.",
    "ConvNeXt encoder and PixelShuffle decoder are not instantiated; SNeRV backbone behavior is simulated via Haar + palette/MGF.",
    "Full 300–1800 epoch training on RTX 4090 requires upstream NeR-SC release when published.",
)
