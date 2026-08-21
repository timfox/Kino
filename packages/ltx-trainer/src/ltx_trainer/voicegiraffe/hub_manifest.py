"""HuggingFace-style dataset manifest for VOICEGIRAFFE (metadata-only, no download)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer.voicegiraffe.config import VoiceGiraffeConfig
from ltx_trainer.voicegiraffe.hub_loader import export_qa_jsonl, export_recording_manifest
from ltx_trainer.voicegiraffe.recording_registry import build_recording_catalog, load_full_qa_pool


def hub_dataset_card(cfg: VoiceGiraffeConfig | None = None) -> dict[str, Any]:
    """Dataset card fields mirroring a Hub README front-matter subset."""
    cfg = cfg or VoiceGiraffeConfig()
    return {
        "repo_id": cfg.hub_repo_id,
        "license": cfg.hub_license,
        "task_categories": ["audio-text-to-text", "question-answering"],
        "language": list(cfg.languages),
        "pretty_name": "VOICEGIRAFFE",
        "paper": cfg.paper_arxiv,
        "size_categories": ["100<n<1K"],
        "dataset_info": {
            "features": {
                "recording_id": "string",
                "audio": "Audio(path, sampling_rate=48000)",
                "qa_jsonl": "string",
                "domain": "string",
                "language": "string",
                "duration_min": "float32",
            },
            "splits": {
                "train": {"num_examples": cfg.n_recordings, "description": "123 native long-form recordings"},
                "qa": {"num_examples": cfg.n_qa_total, "description": "1,500 MC QA items"},
            },
        },
    }


def build_hub_manifest(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    """In-memory manifest describing local export layout (no network I/O)."""
    cfg = cfg or VoiceGiraffeConfig()
    catalog = build_recording_catalog(cfg, seed=seed)
    qa = load_full_qa_pool(cfg, seed=seed)
    return {
        "repo_id": cfg.hub_repo_id,
        "revision": cfg.hub_revision,
        "files": {
            "recordings.jsonl": {"rows": len(catalog), "sha256": "local-export"},
            "qa.jsonl": {"rows": len(qa), "sha256": "local-export"},
            "audio/": {"pattern": "rec-{domain}-{idx}.wav", "count": len(catalog)},
        },
        "splits": hub_dataset_card(cfg)["dataset_info"]["splits"],
        "download_policy": "metadata_only_stub",
    }


def export_hub_bundle(out_dir: str | Path, cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> Path:
    """Write recordings + QA JSONL + dataset card JSON under out_dir."""
    cfg = cfg or VoiceGiraffeConfig()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    catalog = build_recording_catalog(cfg, seed=seed)
    qa = load_full_qa_pool(cfg, seed=seed)
    export_recording_manifest(root / "recordings.jsonl", cfg, seed=seed)
    export_qa_jsonl(root / "qa.jsonl", qa)
    card = {"dataset_card": hub_dataset_card(cfg), "manifest": build_hub_manifest(cfg, seed=seed)}
    (root / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return root


def hub_manifest_smoke(cfg: VoiceGiraffeConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or VoiceGiraffeConfig()
    manifest = build_hub_manifest(cfg, seed=seed)
    card = hub_dataset_card(cfg)
    return {
        "repo_id": manifest["repo_id"],
        "n_recordings": manifest["files"]["recordings.jsonl"]["rows"],
        "n_qa": manifest["files"]["qa.jsonl"]["rows"],
        "metadata_only": manifest["download_policy"] == "metadata_only_stub",
        "splits_match_paper": card["dataset_info"]["splits"]["qa"]["num_examples"] == cfg.n_qa_total,
    }
