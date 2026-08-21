"""Source-level MIP split constraints stub (§2.6)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from ltx_trainer.mygardenbird.config import MygardenbirdConfig

SplitName = Literal["train", "val", "test"]


def greedy_source_split(
    clip_rows: list[dict[str, str]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, SplitName]:
    """Group clips by source_id; assign whole sources to one partition."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for row in clip_rows:
        by_source[row["source_id"]].append(row["file_id"])

    sources = sorted(by_source)
    n = len(sources)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(0, n - n_train - n_val)

    assignment: dict[str, SplitName] = {}
    for i, src in enumerate(sources):
        if i < n_train:
            part: SplitName = "train"
        elif i < n_train + n_val:
            part = "val"
        else:
            part = "test"
        for fid in by_source[src]:
            assignment[fid] = part
    return assignment


def verify_no_source_leakage(
    assignment: dict[str, SplitName],
    clip_rows: list[dict[str, str]],
) -> bool:
    source_to_splits: dict[str, set[str]] = defaultdict(set)
    file_to_source = {r["file_id"]: r["source_id"] for r in clip_rows}
    for fid, split in assignment.items():
        source_to_splits[file_to_source[fid]].add(split)
    return all(len(splits) == 1 for splits in source_to_splits.values())


def split_demo(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MygardenbirdConfig()
    rows = []
    for sp in range(cfg.n_species):
        for clip_idx in range(cfg.clips_per_species):
            src = f"{1000 + (clip_idx % 20)}"
            rows.append(
                {
                    "file_id": f"xc{src}_{clip_idx * 1000}",
                    "source_id": src,
                    "species_idx": str(sp),
                }
            )
    assignment = greedy_source_split(rows, train_ratio=cfg.split_train, val_ratio=cfg.split_val)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    return {
        "n_clips": len(rows),
        "partition_counts": counts,
        "no_source_leakage": verify_no_source_leakage(assignment, rows),
        "target_train": cfg.train_clips,
    }
