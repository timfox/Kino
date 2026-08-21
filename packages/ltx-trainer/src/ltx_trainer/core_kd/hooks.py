"""LTX integration: preprocess meta, caption hints, manifest MER QA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.core_kd.prompts import CORE_KD_CAPTION_HINT, LTX_VIDEO_PROMPT_SUFFIX


def core_kd_enabled() -> bool:
    return os.environ.get("GOPEX_CORE_KD", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not core_kd_enabled():
        return []
    return [CORE_KD_CAPTION_HINT]


def core_kd_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str] | None:
    _ = meta
    lines = caption_hint_lines()
    return lines if lines else None


def ltx_video_prompt_suffix() -> str:
    if not core_kd_enabled():
        return ""
    return LTX_VIDEO_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if core_kd_enabled():
        out.update(core_kd_preprocess_extra())
    return out


def core_kd_preprocess_extra() -> dict[str, Any]:
    return {
        "core_kd": {
            "enabled": True,
            "task": "conversational_mer_missing_modality",
            "arxiv": "2605.29590",
            "method": "CSA + NCE complete-view distillation",
            "components": ["complete_view_state_anchoring", "nonverbal_conflict_exposure"],
        }
    }


def attach_core_kd_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(core_kd_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def _mer_keywords(caption: str) -> dict[str, bool]:
    c = caption.lower()
    return {
        "emotion_cue": any(w in c for w in ("happy", "sad", "angry", "neutral", "frustrat", "excit")),
        "dialogue_cue": any(w in c for w in ("conversation", "dialogue", "speaker", "utterance", "turn")),
        "modality_conflict": any(w in c for w in ("mismatch", "conflict", "disagree", "asynchron")),
        "missing_modality": any(w in c for w in ("missing", "unavailable", "silent", "no audio", "no video")),
    }


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    root = Path(manifest_dir).expanduser().resolve()
    manifest = root / "ltx_manifest.jsonl"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing ltx_manifest.jsonl", "path": str(manifest)}
    out_path = root / "core_kd_qa.jsonl"
    n = 0
    with manifest.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cap = str(row.get("caption") or row.get("prompt") or "")
            cues = _mer_keywords(cap)
            row["core_kd"] = {
                **cues,
                "mer_robust_ready": any(cues.values()),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "rows": n, "output": str(out_path)}
