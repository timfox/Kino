"""LTX integration: preprocess meta, caption hints, manifest NAS QA."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.pesd_vit.prompts import LTX_VIDEO_PROMPT_SUFFIX, PESD_VIT_CAPTION_HINT


def pesd_vit_enabled() -> bool:
    return os.environ.get("GOPEX_PESD_VIT", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not pesd_vit_enabled():
        return []
    return [PESD_VIT_CAPTION_HINT]


def pesd_vit_user_prompt_lines(meta: dict[str, Any] | None = None) -> list[str] | None:
    _ = meta
    lines = caption_hint_lines()
    return lines if lines else None


def ltx_video_prompt_suffix() -> str:
    if not pesd_vit_enabled():
        return ""
    return LTX_VIDEO_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if pesd_vit_enabled():
        out.update(pesd_vit_preprocess_extra())
    return out


def pesd_vit_preprocess_extra() -> dict[str, Any]:
    return {
        "pesd_vit": {
            "enabled": True,
            "task": "nafld_nas_multitask_histology",
            "arxiv": "2605.29852",
            "method": "subspace-decoupled Swin-T + task Adapters + L_ortho",
            "tasks": ["steatosis", "ballooning", "inflammation"],
        }
    }


def attach_pesd_vit_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(pesd_vit_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def _nas_keywords(caption: str) -> dict[str, bool]:
    c = caption.lower()
    return {
        "steatosis_cue": any(w in c for w in ("steatosis", "lipid", "vacuole", "fatty")),
        "ballooning_cue": any(w in c for w in ("balloon", "swollen", "hepatocyte")),
        "inflammation_cue": any(w in c for w in ("inflamm", "foci", "immune", "lobular")),
    }


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    root = Path(manifest_dir).expanduser().resolve()
    manifest = root / "ltx_manifest.jsonl"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing ltx_manifest.jsonl", "path": str(manifest)}
    out_path = root / "pesd_vit_qa.jsonl"
    n = 0
    with manifest.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cap = str(row.get("caption") or row.get("prompt") or "")
            cues = _nas_keywords(cap)
            row["pesd_vit"] = {
                **cues,
                "nas_multitask_ready": any(cues.values()),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "rows": n, "output": str(out_path)}
