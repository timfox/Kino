"""Load benchmark cases from upstream ``annotation.json`` checkouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ltx_trainer.longav_compass.annotation import BenchmarkCase, EventAnnotation, case_to_dict


def _event_from_dict(raw: dict[str, Any]) -> EventAnnotation:
    return EventAnnotation(
        event_id=int(raw.get("event_id", raw.get("id", 0))),
        time_range=str(raw.get("time_range", raw.get("time", "0:00-0:05"))),
        action=str(raw.get("action", "")),
        completion_flag=str(raw.get("completion_flag", raw.get("completion", "complete"))),
        visual_description=str(raw.get("visual_description", raw.get("visual", ""))),
        audio_expectation=str(raw.get("audio_expectation", raw.get("audio", ""))),
        qa_questions=list(raw.get("qa_questions", raw.get("questions", [])) or []),
    )


def case_from_dict(data: dict[str, Any]) -> BenchmarkCase:
    """Parse a case dict (upstream JSON or ``case_to_dict`` export)."""
    events_raw = data.get("events") or data.get("event_annotations") or []
    return BenchmarkCase(
        case_id=str(data["case_id"]),
        task=str(data.get("task", "T2AV")).upper(),
        scenario=str(data.get("scenario", "Performance Ads")),
        complexity=str(data.get("complexity", "L4")),
        language=str(data.get("language", "en")),
        global_description=str(
            data.get("global_description", data.get("global_prompt", data.get("description", "")))
        ),
        events=[_event_from_dict(e) for e in events_raw],
        reference_image=data.get("reference_image"),
        reference_video=data.get("reference_video"),
        identity_tracking=list(data.get("identity_tracking", []) or []),
        physical_constraints=list(data.get("physical_constraints", []) or []),
    )


def load_case_json(path: str | Path) -> BenchmarkCase:
    """Load one case from ``annotation.json``."""
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {p}")
    return case_from_dict(data)


def case_annotation_path(checkout_root: str | Path, case_id: str) -> Path:
    return Path(checkout_root) / "cases" / case_id / "annotation.json"


def discover_case_ids(checkout_root: str | Path) -> list[str]:
    """List case IDs under ``<root>/cases/*/annotation.json``."""
    root = Path(checkout_root)
    cases_dir = root / "cases"
    if not cases_dir.is_dir():
        return []
    ids: list[str] = []
    for child in sorted(cases_dir.iterdir()):
        if child.is_dir() and (child / "annotation.json").is_file():
            ids.append(child.name)
    return ids


def load_case_from_upstream(checkout_root: str | Path, case_id: str) -> BenchmarkCase:
    path = case_annotation_path(checkout_root, case_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing annotation: {path}")
    return load_case_json(path)


def model_output_paths(
    checkout_root: str | Path,
    case_id: str,
    model_slug: str,
) -> dict[str, Path]:
    """Expected per-model artefact paths (paper reproducibility layout)."""
    base = Path(checkout_root) / "results" / model_slug / case_id
    return {
        "full_video": base / "full_video.mp4",
        "canonical_events": base / "canonical_events.json",
        "events_dir": base / "events",
        "boundaries_dir": base / "boundaries",
    }


def summarize_checkout(checkout_root: str | Path) -> dict[str, Any]:
    """Inventory an upstream checkout without evaluating."""
    root = Path(checkout_root)
    case_ids = discover_case_ids(root)
    models: list[str] = []
    results_dir = root / "results"
    if results_dir.is_dir():
        models = sorted(d.name for d in results_dir.iterdir() if d.is_dir())
    return {
        "root": str(root),
        "case_count": len(case_ids),
        "case_ids_sample": case_ids[:8],
        "model_slugs": models,
        "has_cases_dir": (root / "cases").is_dir(),
        "has_results_dir": results_dir.is_dir(),
    }
