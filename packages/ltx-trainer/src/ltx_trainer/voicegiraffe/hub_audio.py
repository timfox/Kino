"""Synthetic hour-scale audio manifest + deterministic wave loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig


def build_audio_manifest(
    catalog: list[dict[str, Any]],
    *,
    root: str = "voicegiraffe/audio",
    sample_rate_hz: float = 48000.0,
) -> list[dict[str, Any]]:
    """Hub-style manifest rows (paths only; waves synthesized locally)."""
    rows: list[dict[str, Any]] = []
    for rec in catalog:
        rec_id = rec["recording_id"]
        rows.append(
            {
                "recording_id": rec_id,
                "path": f"{root}/{rec_id}.wav",
                "duration_min": rec["duration_min"],
                "sample_rate_hz": sample_rate_hz,
                "domain": rec["domain"],
                "language": rec["language"],
            }
        )
    return rows


def export_audio_manifest(path: str | Path, catalog: list[dict[str, Any]], *, root: str = "voicegiraffe/audio") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = build_audio_manifest(catalog, root=root)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


def load_audio_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def synthesize_recording_wave(
    recording: dict[str, Any],
    *,
    sample_rate_hz: float = 48000.0,
    seed: int = 0,
    speed_factor: float = 0.01,
) -> np.ndarray:
    """Deterministic pseudo-audio keyed by recording_id (short proxy for CPU smoke)."""
    rng = np.random.default_rng(seed + hash(recording["recording_id"]) % 2**31)
    n = max(1, int(sample_rate_hz * recording["duration_min"] * 60 * speed_factor))
    t = np.arange(n) / sample_rate_hz
    f0 = 220.0 + 30.0 * (hash(recording["domain"]) % 5)
    wave = 0.2 * np.sin(2 * np.pi * f0 * t)
    wave += 0.05 * rng.standard_normal(n)
    if recording.get("language") == "ZH":
        wave *= 1.05
    return wave.astype(np.float64)


def load_recording_waves(
    catalog: list[dict[str, Any]],
    *,
    sample_rate_hz: float = 48000.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    return {
        rec["recording_id"]: synthesize_recording_wave(rec, sample_rate_hz=sample_rate_hz, seed=seed)
        for rec in catalog
    }


def hub_audio_smoke(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    from ltx_trainer.voicegiraffe.recording_registry import build_recording_catalog

    cfg = cfg or VoiceGiraffeConfig()
    catalog = build_recording_catalog(cfg, seed=seed)[:8]
    manifest = build_audio_manifest(catalog)
    waves = load_recording_waves(catalog, seed=seed)
    total_samples = sum(w.size for w in waves.values())
    return {
        "n_manifest_rows": len(manifest),
        "n_waves": len(waves),
        "total_samples": int(total_samples),
        "sample_rate_hz": 48000.0,
    }
