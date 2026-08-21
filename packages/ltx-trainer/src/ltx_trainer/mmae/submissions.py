"""MMAE submission layout: predictions JSON (paper format) + coverage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer.mmae.paths import resolve_prediction_wav
from ltx_trainer.mmae.sample import MMAESample


def prediction_wav_relpath(
    predictions_dir: Path,
    sample_id: str,
    *,
    audio_root: Path | None = None,
) -> str | None:
    """Resolved output WAV as path relative to *audio_root* (default: predictions parent)."""
    wav = resolve_prediction_wav(predictions_dir, sample_id)
    if wav is None:
        return None
    root = audio_root or predictions_dir.parent
    try:
        return str(wav.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(wav.resolve())


def build_prediction_entry(
    sample: MMAESample,
    predictions_dir: Path,
    *,
    audio_root: Path | None = None,
) -> dict[str, Any] | None:
    """One paper-style row: user messages + assistant audio turn."""
    rel = prediction_wav_relpath(predictions_dir, sample.sample_id, audio_root=audio_root)
    if rel is None:
        return None
    base = sample.to_dict()
    messages = list(base.get("messages", []))
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "audio", "audio_url": rel}],
        }
    )
    return {
        "id": sample.sample_id,
        "complexity": sample.complexity,
        "modality": sample.modality,
        "granularity": sample.granularity,
        "operations": sample.operations,
        "messages": messages,
        "tags": sample.tags,
        "rubrics": [r.to_dict() for r in sample.rubrics],
    }


def build_predictions_json(
    samples: list[MMAESample],
    predictions_dir: str | Path,
    output_path: str | Path,
    *,
    audio_root: str | Path | None = None,
    require_all: bool = False,
) -> dict[str, Any]:
    """Write upstream ``eval.score``-compatible predictions JSON."""
    pred_root = Path(predictions_dir)
    out_path = Path(output_path)
    root = Path(audio_root) if audio_root is not None else pred_root.parent
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for sample in samples:
        entry = build_prediction_entry(sample, pred_root, audio_root=root)
        if entry is None:
            missing.append(sample.sample_id)
            continue
        entries.append(entry)
    if require_all and missing:
        raise FileNotFoundError(f"missing predictions for: {missing}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output": str(out_path),
        "audio_root": str(root),
        "predictions_dir": str(pred_root),
        "num_samples": len(samples),
        "num_entries": len(entries),
        "missing": missing,
        "coverage": len(entries) / max(len(samples), 1),
    }
