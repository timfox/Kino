"""PEMark: position-encoding watermarking for API responses (arXiv:2605.21865)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PEMarkConfig:
    paper_arxiv: str = "arXiv:2605.21865"

    # Watermark bit-length used in figures/examples (paper uses variable L; examples cite 64-bit).
    watermark_bits_L: int = 64

    # Group size threshold T (keys per group). Capacity condition: 2^L <= T!
    threshold_T: int = 21  # for L=64, paper states T=21

    # Majority voting across groups
    min_groups_for_vote: int = 3

    # Proxy gateway notes (OpenResty in paper) — stub only
    gateway_name: str = "proxy_gateway"
    gateway_stack: str = "OpenResty (paper); stub only"

