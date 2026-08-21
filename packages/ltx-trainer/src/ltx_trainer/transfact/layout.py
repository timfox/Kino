"""TransFACT architecture notes and stage taxonomy."""

from __future__ import annotations

STAGE_CLASSES: tuple[str, ...] = (
    "1_cell",
    "2_cells",
    "3_cells",
    "4_cells",
    "5_cells",
    "6_cells",
    "7_cells",
    "8_cells",
    "9_plus_cells",
    "cleavage_event",
    "developmental_arrest",
)

ARCHITECTURE_NOTES: tuple[str, ...] = (
    "FACT-inspired dual branch: dilated temporal conv frame branch + transformer stage tokens.",
    "Bidirectional cross-attention between frames and stage tokens across B=3 update blocks.",
    "Global transferability head pools frame + stage features (binary T vs NT at 4 DPI).",
    "Optional MHI cross-attention branch (τ=15, θ=20) for motion cues.",
    "Pre-extracted 2D-CNN frame features; end-to-end training on top of frozen embeddings.",
)

LIMITATIONS: tuple[str, ...] = (
    "Single INRAE-Gertrude-DT bovine dataset; no public Hub release in paper.",
    "MHI-only input underperforms frame features; motion branch is supplementary.",
    "vs SFR comparison uses Wilcoxon p-values trending but not always p<0.05.",
    "Manual stage labels required for auxiliary supervision during training.",
)
