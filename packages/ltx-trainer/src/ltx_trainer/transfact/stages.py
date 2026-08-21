"""Embryo developmental stage labeling (toy)."""

from __future__ import annotations

STAGE_NAMES: tuple[str, ...] = (
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


def cell_count_to_stage(n_cells: int, *, arrested: bool = False) -> str:
    if arrested:
        return "developmental_arrest"
    if n_cells <= 0:
        return "1_cell"
    if n_cells >= 9:
        return "9_plus_cells"
    return STAGE_NAMES[n_cells - 1]


def stage_frame_labels(cell_counts: list[int], *, arrested_at: int | None = None) -> list[str]:
    labels = [cell_count_to_stage(n) for n in cell_counts]
    if arrested_at is not None and 0 <= arrested_at < len(labels):
        labels[arrested_at] = "developmental_arrest"
    return labels
