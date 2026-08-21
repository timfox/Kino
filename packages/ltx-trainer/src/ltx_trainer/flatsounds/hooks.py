"""LTX integration: preprocess meta, caption hints, manifest V2A physical QA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.flatsounds.prompts import FLATSOUNDS_CAPTION_HINT, PROMPT_SUFFIX


def flatsounds_enabled() -> bool:
    return os.environ.get("GOPEX_FLATSOUNDS", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not flatsounds_enabled():
        return []
    return [FLATSOUNDS_CAPTION_HINT]


def flatsounds_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str] | None:
    _ = meta
    lines = caption_hint_lines()
    return lines if lines else None


def ltx_video_prompt_suffix() -> str:
    if not flatsounds_enabled():
        return ""
    return PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if flatsounds_enabled():
        out.update(flatsounds_preprocess_extra())
    return out


def flatsounds_preprocess_extra() -> dict[str, Any]:
    return {
        "flatsounds": {
            "enabled": True,
            "task": "v2a_physical_benchmark",
            "arxiv": "2605.30339",
            "method": "counterfactual pairs + single-video physics metrics",
            "modes": ["alignment", "physical_correctness"],
        }
    }


def attach_flatsounds_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(flatsounds_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def _impact_keywords(caption: str) -> dict[str, bool]:
    c = caption.lower()
    return {
        "impact_cue": any(w in c for w in ("tap", "hit", "strike", "clap", "knock", "impact")),
        "material_cue": any(w in c for w in ("metal", "wood", "glass", "ceramic", "plastic")),
        "room_cue": any(w in c for w in ("hall", "stair", "reverb", "echo", "room", "corridor")),
        "timing_cue": any(w in c for w in ("onset", "rhythm", "beat", "twice", "repeated")),
    }


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    root = Path(manifest_dir).expanduser().resolve()
    manifest = root / "ltx_manifest.jsonl"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing ltx_manifest.jsonl", "path": str(manifest)}
    out_path = root / "flatsounds_qa.jsonl"
    n = 0
    with manifest.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cap = str(row.get("caption") or row.get("prompt") or "")
            cues = _impact_keywords(cap)
            row["flatsounds"] = {
                **cues,
                "v2a_physics_ready": any(cues.values()),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "rows": n, "output": str(out_path)}
