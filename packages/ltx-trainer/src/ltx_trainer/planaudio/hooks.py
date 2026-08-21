"""LTX integration: preprocess meta, caption hints, manifest scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.planaudio.prompts import LTX_AV_PROMPT_SUFFIX, PLANAUDIO_CAPTION_HINT
from ltx_trainer.planaudio.scenarios import SCENARIO_COMPOSITE, classify_free_form_prompt
from ltx_trainer.planaudio.scoring import semantic_coverage_factor


def planaudio_enabled() -> bool:
    return os.environ.get("GOPEX_PLANAUDIO", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not planaudio_enabled():
        return []
    return [PLANAUDIO_CAPTION_HINT]


def planaudio_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str]:
    """Extra Gemma caption instructions when ``GOPEX_PLANAUDIO=1``."""
    _ = meta
    return caption_hint_lines()


def ltx_av_prompt_suffix() -> str:
    if not planaudio_enabled():
        return ""
    return LTX_AV_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if planaudio_enabled():
        out.update(planaudio_preprocess_extra())
    return out


def planaudio_preprocess_extra() -> dict[str, Any]:
    return {
        "planaudio": {
            "enabled": True,
            "task": "free_form_text_to_unified_audio",
            "arxiv": "2605.28063",
            "scenarios": ["sound", "speech", "composite"],
            "latent_cot_steps": 6,
            "backbone": "Qwen2.5-1.5B",
        }
    }


def attach_planaudio_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(planaudio_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def score_prompt_composition(caption: str) -> dict[str, Any]:
    """Tag manifest captions with scenario + SCF proxy for dataset QA."""
    scenario = classify_free_form_prompt(caption)
    # Toy event similarities from keyword coverage
    events = []
    lower = caption.lower()
    if '"' in caption or "says" in lower:
        events.append(0.85 if '"' in caption else 0.55)
    for cue in ("music", "applause", "guitar", "rain", "laugh"):
        if cue in lower:
            events.append(0.75)
    scf = semantic_coverage_factor(events)
    composite_ready = scenario == SCENARIO_COMPOSITE or (len(events) >= 2)
    return {
        "scenario": scenario,
        "semantic_coverage_factor_proxy": round(scf, 4),
        "composite_ready": composite_ready,
        "caption_chars": len(caption),
    }


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    man = Path(manifest_dir).expanduser().resolve()
    ds = man / "dataset.json"
    if not ds.is_file():
        return {"ok": False, "reason": "dataset.json not found", "path": str(ds)}
    rows = json.loads(ds.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return {"ok": False, "reason": "dataset.json must be a list"}
    report_path = man / "planaudio_qa.jsonl"
    counts = {"sound": 0, "speech": 0, "composite": 0}
    scored: list[dict[str, Any]] = []
    for row in rows:
        cap = str(row.get("caption") or row.get("prompt") or "")
        s = score_prompt_composition(cap)
        counts[s["scenario"]] = counts.get(s["scenario"], 0) + 1
        scored.append({"media_path": row.get("media_path"), **s})
    with report_path.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(scored)
    return {
        "ok": True,
        "scored": n,
        "scenario_counts": counts,
        "composite_fraction": round(counts.get("composite", 0) / n, 4) if n else 0.0,
        "report": str(report_path),
    }
