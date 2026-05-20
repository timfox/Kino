"""Metadata written beside ``.precomputed`` so training can match caption preprocess settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREPROCESS_META_FILENAME = "preprocess_meta.json"


def write_preprocess_meta(
    output_base: Path,
    *,
    model_path: str | Path,
    text_encoder_path: str | Path,
    flat_dim_bridge_rank: int | None,
    dataset_file: str | Path | None = None,
    resolution_buckets: list[tuple[int, int, int]] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``preprocess_meta.json`` under the precomputed root."""
    output_base = Path(output_base).expanduser().resolve()
    payload: dict[str, Any] = {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "text_encoder_path": str(Path(text_encoder_path).expanduser().resolve()),
        "flat_dim_bridge_rank": flat_dim_bridge_rank,
        "bridge_kind": (
            f"low_rank_{flat_dim_bridge_rank}"
            if flat_dim_bridge_rank is not None
            else "dense_random"
        ),
    }
    if dataset_file is not None:
        payload["dataset_file"] = str(Path(dataset_file).expanduser().resolve())
    if resolution_buckets is not None:
        payload["resolution_buckets"] = [f"{w}x{h}x{f}" for w, h, f in resolution_buckets]
    if extra:
        payload.update(extra)
    path = output_base / PREPROCESS_META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_preprocess_meta(preprocessed_root: Path) -> dict[str, Any] | None:
    root = Path(preprocessed_root).expanduser().resolve()
    for candidate in (root, root / ".precomputed", root.parent / ".precomputed"):
        path = candidate / PREPROCESS_META_FILENAME
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None
