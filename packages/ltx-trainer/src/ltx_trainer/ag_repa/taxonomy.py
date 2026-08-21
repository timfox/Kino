"""Selection strategies and SCD taxonomy (paper Sec. 5.3)."""

from __future__ import annotations

from enum import Enum


class SelectionStrategy(str, Enum):
    NONE = "base_no_layer_align"
    RANDOM = "random_layers"
    HIGHEST_LASP = "highest_lasp"
    GRADIENT_NORM = "gradient_norm"
    HIGHEST_FOG_A = "highest_fog_a"
    STATIC_REPA_L4 = "repa_layer_4"
    STATIC_REPA_L8 = "repa_layer_8"
    STATIC_REPA_L12 = "repa_layer_12"
    STATIC_REPA_MULTI = "repa_l4_l8_l12"
    DEEP_REPA = "repa_deep_l20_22"
    SHALLOW_REPA = "repa_shallow_l1_3"
    AG_REPA = "ag_repa_top_k"


STRATEGY_LABELS: dict[SelectionStrategy, str] = {
    SelectionStrategy.NONE: "Base (no layer align.)",
    SelectionStrategy.RANDOM: "Random Control",
    SelectionStrategy.HIGHEST_LASP: "Highest LASP",
    SelectionStrategy.GRADIENT_NORM: "Gradient Norm",
    SelectionStrategy.HIGHEST_FOG_A: "Highest FoG-A (Ours)",
    SelectionStrategy.AG_REPA: "AG-REPA (Top-3)",
    SelectionStrategy.STATIC_REPA_L4: "REPA @ Layer 4",
    SelectionStrategy.STATIC_REPA_L8: "REPA @ Layer 8",
    SelectionStrategy.STATIC_REPA_L12: "REPA @ Layer 12",
    SelectionStrategy.STATIC_REPA_MULTI: "REPA @ L4, 8, 12",
    SelectionStrategy.DEEP_REPA: "REPA @ Deep (L20–L22)",
    SelectionStrategy.SHALLOW_REPA: "REPA @ Shallow (L1–L3)",
}
