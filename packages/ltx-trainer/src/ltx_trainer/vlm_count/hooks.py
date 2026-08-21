"""LTX integration: preprocess meta, caption hints, manifest counting QA."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ltx_trainer.vlm_count.prompts import LTX_VIDEO_PROMPT_SUFFIX, VLM_COUNT_CAPTION_HINT


def vlm_count_enabled() -> bool:
    return os.environ.get("GOPEX_VLM_COUNT", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not vlm_count_enabled():
        return []
    return [VLM_COUNT_CAPTION_HINT]


def vlm_count_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str] | None:
    _ = meta
    lines = caption_hint_lines()
    return lines if lines else None


def ltx_video_prompt_suffix() -> str:
    if not vlm_count_enabled():
        return ""
    return LTX_VIDEO_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if vlm_count_enabled():
        out.update(vlm_count_preprocess_extra())
    return out


def vlm_count_preprocess_extra() -> dict[str, Any]:
    return {
        "vlm_count": {
            "enabled": True,
            "task": "visual_counting_bottleneck",
            "arxiv": "2605.30170",
            "stages": [
                "visual_individuation",
                "magnitude_awareness",
                "symbolic_mapping",
            ],
            "hypothesis": "fractured_magnitude",
        }
    }


def attach_vlm_count_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(vlm_count_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


_COUNT_RE = re.compile(
    r"\b(\d+)\s+(stones?|pieces?|objects?|people|items|black|white)\b|\bcount(?:ing)?\s+(\d+)\b",
    re.I,
)


def _counting_keywords(caption: str) -> dict[str, bool]:
    c = caption.lower()
    m = _COUNT_RE.search(caption)
    n_hint = int(m.group(1) or m.group(3)) if m else None
    return {
        "counting_cue": any(w in c for w in ("count", "how many", "number of", "total")),
        "enumerable_objects": any(w in c for w in ("stone", "piece", "object", "person", "item")),
        "grid_or_board": any(w in c for w in ("board", "grid", "go ", "chess", "checker")),
        "has_numeric_hint": n_hint is not None,
        "numeric_hint": n_hint,
    }


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    root = Path(manifest_dir).expanduser().resolve()
    manifest = root / "ltx_manifest.jsonl"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing ltx_manifest.jsonl", "path": str(manifest)}
    out_path = root / "vlm_count_qa.jsonl"
    rows = []
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cap = str(rec.get("caption") or rec.get("text") or "")
            cues = _counting_keywords(cap)
            rows.append({"id": rec.get("id"), "caption_cues": cues})
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    n_count = sum(1 for r in rows if r["caption_cues"]["counting_cue"])
    return {
        "ok": True,
        "manifest": str(manifest),
        "qa_path": str(out_path),
        "n_rows": len(rows),
        "n_counting_captions": n_count,
    }
