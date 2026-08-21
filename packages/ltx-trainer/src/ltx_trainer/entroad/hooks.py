"""LTX integration: preprocess meta, caption hints, manifest scoring."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ltx_trainer.entroad.prompts import ENTROAD_CAPTION_HINT, LTX_VIDEO_PROMPT_SUFFIX


def entroad_enabled() -> bool:
    return os.environ.get("GOPEX_ENTROAD", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not entroad_enabled():
        return []
    return [ENTROAD_CAPTION_HINT]


def entroad_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str]:
    _ = meta
    return caption_hint_lines()


def ltx_video_prompt_suffix() -> str:
    if not entroad_enabled():
        return ""
    return LTX_VIDEO_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if entroad_enabled():
        out.update(entroad_preprocess_extra())
    return out


def entroad_preprocess_extra() -> dict[str, Any]:
    return {
        "entroad": {
            "enabled": True,
            "task": "zero_shot_anomaly_detection",
            "arxiv": "2605.28630",
            "backbone": "CLIP ViT-L/14 + structural entropy router + dual-branch prompts",
            "train_source": "MVTec-AD",
            "test_count": 10,
        }
    }


def attach_entroad_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(entroad_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


_LOCALIZED = re.compile(
    r"\b(scratch|crack|chip|dent|missing|broken|hole|stain|spot|defect|chip)\b", re.I
)
_DIFFUSE = re.compile(
    r"\b(diffuse|irregular|blurry|smear|spread|patchy|uneven|discolor|abnormal texture)\b", re.I
)


def score_anomaly_caption(caption: str) -> dict[str, Any]:
    """Tag captions for ZSAD-oriented QA (localized vs diffuse cues)."""
    loc = len(_LOCALIZED.findall(caption))
    dif = len(_DIFFUSE.findall(caption))
    branch_hint = "branch_a" if loc >= dif else "branch_b"
    return {
        "localized_cue_count": loc,
        "diffuse_cue_count": dif,
        "branch_prior_hint": branch_hint,
        "anomaly_caption_ready": (loc + dif) >= 1,
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
    report_path = man / "entroad_qa.jsonl"
    ready = 0
    scored: list[dict[str, Any]] = []
    for row in rows:
        cap = str(row.get("caption") or row.get("prompt") or "")
        s = score_anomaly_caption(cap)
        if s["anomaly_caption_ready"]:
            ready += 1
        scored.append({"media_path": row.get("media_path"), **s})
    with report_path.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(scored)
    return {
        "ok": True,
        "scored": n,
        "anomaly_ready_fraction": round(ready / n, 4) if n else 0.0,
        "report": str(report_path),
    }
