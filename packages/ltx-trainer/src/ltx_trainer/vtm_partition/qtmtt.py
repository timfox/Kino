"""QTMTT split modes and combinatorial scale facts (VVC vs HEVC)."""

from __future__ import annotations

from typing import Final

SPLIT_NS: Final[str] = "NS"
SPLIT_QT: Final[str] = "QT"
SPLIT_BTH: Final[str] = "BTH"
SPLIT_BTV: Final[str] = "BTV"
SPLIT_TTH: Final[str] = "TTH"
SPLIT_TTV: Final[str] = "TTV"

ALL_SPLITS: Final[tuple[str, ...]] = (
    SPLIT_NS,
    SPLIT_QT,
    SPLIT_BTH,
    SPLIT_BTV,
    SPLIT_TTH,
    SPLIT_TTV,
)

# Number of sub-CUs produced by each split (paper Sec. 1).
SUB_CU_COUNT: Final[dict[str, int]] = {
    SPLIT_NS: 1,
    SPLIT_QT: 4,
    SPLIT_BTH: 2,
    SPLIT_BTV: 2,
    SPLIT_TTH: 3,
    SPLIT_TTV: 3,
}

# Orientation flag HV in split-series encoding: 0=QT, +1=horizontal, -1=vertical.
SPLIT_HV: Final[dict[str, int]] = {
    SPLIT_NS: 0,
    SPLIT_QT: 0,
    SPLIT_BTH: 1,
    SPLIT_BTV: -1,
    SPLIT_TTH: 1,
    SPLIT_TTV: -1,
}


def hevc_vs_vvc_scale_facts() -> dict[str, int | float]:
    """Worst-case intra 64×64 CTU partitioning scale (paper Introduction)."""
    return {
        "hevc_max_blocks_64x64": 341,
        "hevc_pixels_per_mode_k": 20_000,
        "vvc_max_blocks_64x64": 721_000,
        "vvc_pixels_per_mode_m": 19_000_000,
        "block_ratio_vvc_over_hevc": 721_000 / 341,
        "pixel_ratio_vvc_over_hevc": 19_000_000 / 20_000,
    }
