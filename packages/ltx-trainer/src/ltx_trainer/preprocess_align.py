"""Compare training YAML / form settings with ``preprocess_meta.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ltx_trainer.preprocess_meta import read_preprocess_meta


def check_preprocess_training_alignment(
    preprocessed_root: str | Path,
    *,
    model_path: str | Path | None = None,
    text_encoder_path: str | Path | None = None,
    flat_dim_bridge_rank: int | None = None,
) -> list[str]:
    """Return human-readable mismatch messages (empty if aligned or no meta)."""
    meta = read_preprocess_meta(Path(preprocessed_root))
    if meta is None:
        return ["No preprocess_meta.json — re-run process_dataset.py after upgrading ltx-trainer."]

    issues: list[str] = []
    if meta.get("bridge_kind") == "dense_random":
        issues.append(
            "Conditions were embedded with the dense random bridge. "
            "Re-run process_dataset with --flat-dim-bridge-rank 32."
        )

    meta_rank = meta.get("flat_dim_bridge_rank")
    if flat_dim_bridge_rank is not None and meta_rank != flat_dim_bridge_rank:
        issues.append(f"flat_dim_bridge_rank: preprocess={meta_rank!r} training={flat_dim_bridge_rank!r}")

    if model_path is not None:
        mp = str(Path(model_path).expanduser().resolve())
        if meta.get("model_path") and meta.get("model_path") != mp:
            issues.append("model_path differs between preprocess_meta.json and training config.")

    if text_encoder_path is not None:
        ep = str(Path(text_encoder_path).expanduser().resolve())
        if meta.get("text_encoder_path") and meta.get("text_encoder_path") != ep:
            issues.append("text_encoder_path differs between preprocess_meta.json and training config.")

    return issues
