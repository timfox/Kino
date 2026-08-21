"""LTX integration: preprocess meta, caption hints, manifest scoring."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ltx_trainer.eigenet.prompts import EIGENET_CAPTION_HINT, LTX_AV_PROMPT_SUFFIX


def eigenet_enabled() -> bool:
    return os.environ.get("GOPEX_EIGENET", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not eigenet_enabled():
        return []
    return [EIGENET_CAPTION_HINT]


def eigenet_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str]:
    _ = meta
    return caption_hint_lines()


def ltx_av_prompt_suffix() -> str:
    if not eigenet_enabled():
        return ""
    return LTX_AV_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if eigenet_enabled():
        out.update(eigenet_preprocess_extra())
    return out


def eigenet_preprocess_extra() -> dict[str, Any]:
    return {
        "eigenet": {
            "enabled": True,
            "task": "few_shot_novel_view_rir",
            "arxiv": "2605.28101",
            "backbone": "CVAT + geometry-informed modulation",
            "reference_counts": [1, 4, 8],
            "sampling_rate_hz": 16000,
            "github": "https://github.com/FEAfeatherTHER/EigeNet",
        }
    }


def attach_eigenet_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(eigenet_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


_REVERB_CUES = re.compile(
    r"\b(reverb|echo|hall|room tone|reflection|dry|wet|spacious|cavernous|intimate|"
    r"reverberant|acoustic|damped|absorbent)\b",
    re.I,
)


def score_spatial_audio_caption(caption: str) -> dict[str, Any]:
    """Tag captions with spatial-acoustic cues for dataset QA."""
    lower = caption.lower()
    cues = [m.group(0).lower() for m in _REVERB_CUES.finditer(caption)]
    n_cues = len(set(cues))
    has_geometry = any(w in lower for w in ("room", "hall", "corridor", "cafe", "classroom", "outdoor"))
    ref_hint = bool(re.search(r"\b(\d+)\s*(reference|ref)\s*view", lower))
    return {
        "spatial_acoustic_cue_count": n_cues,
        "geometry_mentioned": has_geometry,
        "reference_view_hint": ref_hint,
        "few_shot_ready": n_cues >= 1 and has_geometry,
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
    report_path = man / "eigenet_qa.jsonl"
    scored: list[dict[str, Any]] = []
    ready = 0
    for row in rows:
        cap = str(row.get("caption") or row.get("prompt") or "")
        s = score_spatial_audio_caption(cap)
        if s["few_shot_ready"]:
            ready += 1
        scored.append({"media_path": row.get("media_path"), **s})
    with report_path.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(scored)
    return {
        "ok": True,
        "scored": n,
        "few_shot_ready_fraction": round(ready / n, 4) if n else 0.0,
        "report": str(report_path),
    }
