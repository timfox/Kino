"""VOICEGIRAFFE recording catalog and JSONL loader."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.dataset import QAItem, load_qa_items
from ltx_trainer.voicegiraffe.recording_registry import (
    attach_qa_counts,
    build_recording_catalog,
    catalog_summary,
    load_full_qa_pool,
    recording_registry_smoke,
)


def recording_catalog(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> list[dict[str, Any]]:
    """123-recording manifest with per-recording QA counts."""
    cfg = cfg or VoiceGiraffeConfig()
    catalog = build_recording_catalog(cfg, seed=seed)
    qa = load_full_qa_pool(cfg, seed=seed)
    return attach_qa_counts(catalog, qa)


def export_qa_jsonl(path: str | Path, items: list[QAItem] | None = None) -> Path:
    items = items or load_qa_items()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item)) + "\n")
    return p


def export_recording_manifest(path: str | Path, cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = recording_catalog(cfg, seed=seed)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


def load_qa_jsonl(path: str | Path) -> list[QAItem]:
    p = Path(path)
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [QAItem(**row) for row in rows]


def load_recording_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def hub_loader_smoke(*, jsonl_path: str | Path | None = None, cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    catalog = recording_catalog(cfg)
    full_qa = load_full_qa_pool(cfg)
    reg = recording_registry_smoke(cfg)
    summary = catalog_summary(catalog)
    roundtrip_ok = True
    if jsonl_path is not None:
        qa_path = export_qa_jsonl(jsonl_path, full_qa)
        manifest_path = export_recording_manifest(Path(jsonl_path).parent / "recordings.jsonl", cfg)
        roundtrip_ok = len(load_qa_jsonl(qa_path)) == len(full_qa) and len(load_recording_manifest(manifest_path)) == cfg.n_recordings
    return {
        "n_recordings_catalog": summary["n_recordings"],
        "n_qa_full_pool": len(full_qa),
        "n_qa_builtin": len(load_qa_items()),
        "jsonl_roundtrip_ok": roundtrip_ok,
        "domains": summary["domains"],
        "matches_paper_scale": reg["matches_paper_scale"],
        "mean_duration_min": summary["mean_duration_min"],
    }
