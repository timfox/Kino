"""LTX integration: preprocess meta, caption hints, manifest MOV tags."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.cert_las.prompts import CERT_LAS_CAPTION_HINT, LTX_VIDEO_PROMPT_SUFFIX


def cert_las_enabled() -> bool:
    return os.environ.get("GOPEX_CERT_LAS", "0").strip().lower() in ("1", "true", "yes")


def caption_hint_lines() -> list[str]:
    if not cert_las_enabled():
        return []
    return [CERT_LAS_CAPTION_HINT]


def cert_las_user_prompt_lines(meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
    _ = meta
    lines = caption_hint_lines()
    return lines if lines else None


def ltx_video_prompt_suffix() -> str:
    if not cert_las_enabled():
        return ""
    return LTX_VIDEO_PROMPT_SUFFIX


def merge_preprocess_extra(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(extra or {})
    if cert_las_enabled():
        out.update(cert_las_preprocess_extra())
    return out


def cert_las_preprocess_extra() -> dict[str, Any]:
    return {
        "cert_las": {
            "enabled": True,
            "task": "certified_diffusion_mov",
            "arxiv": "2605.29809",
            "method": "layer-adaptive smoothing + diffusion classifier WR/RP",
            "trigger_free": True,
        }
    }


def attach_cert_las_to_preprocess_meta(preprocessed_root: str | Path) -> dict[str, Any]:
    root = Path(preprocessed_root).expanduser().resolve()
    meta_path = root / "preprocess_meta.json"
    if not meta_path.is_file():
        return {"ok": False, "reason": "missing preprocess_meta.json", "path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(cert_las_preprocess_extra())
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(meta_path)}


def qa_manifest_directory(manifest_dir: str | Path) -> dict[str, Any]:
    """Tag manifest rows with MOV provenance readiness (class-prompt neutrality)."""
    root = Path(manifest_dir).expanduser().resolve()
    manifest = root / "ltx_manifest.jsonl"
    if not manifest.is_file():
        return {"ok": False, "reason": "missing ltx_manifest.jsonl", "path": str(manifest)}
    out_path = root / "cert_las_qa.jsonl"
    n = 0
    with manifest.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cap = str(row.get("caption") or row.get("prompt") or "")
            row["cert_las"] = {
                "trigger_free_ok": "trigger" not in cap.lower() and "wm" not in cap.lower(),
                "mov_tag": "certified_provenance_optional",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return {"ok": True, "rows": n, "output": str(out_path)}
