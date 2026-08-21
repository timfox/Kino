"""Resumable execution state for composed LTX3 minute renders."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ltx_trainer.ltx3.render_plan import MinuteRenderPlan, SegmentRenderStep


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def file_ready(path: str | Path | None) -> bool:
    if not path:
        return False
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def state_path_for_plan(plan: MinuteRenderPlan) -> Path:
    return Path(plan.full_output).expanduser().resolve().parent / "render_state.json"


def plan_hash(plan: MinuteRenderPlan) -> str:
    """Stable digest for commands and outputs that make resume decisions safe."""
    payload = {
        "full_output": plan.full_output,
        "segments": [
            {
                "index": s.index,
                "output_mp4": s.output_mp4,
                "hero_in": s.hero_in,
                "hero_out": s.hero_out,
                "frames": s.frames,
                "infer_shell": s.infer_shell,
                "extract_hero_shell": s.extract_hero_shell,
            }
            for s in plan.segments
        ],
        "concat_shell": plan.concat_shell,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _segment_file_complete(seg: SegmentRenderStep) -> bool:
    if not file_ready(seg.output_mp4):
        return False
    return seg.hero_out is None or file_ready(seg.hero_out)


def _init_state(plan: MinuteRenderPlan, *, previous: dict[str, Any] | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "version": 1,
        "plan_hash": plan_hash(plan),
        "out_dir": plan.out_dir,
        "full_output": plan.full_output,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "segments": {},
        "concat": {"status": "pending", "output_mp4": plan.full_output},
    }
    if previous and previous.get("plan_hash") and previous.get("plan_hash") != state["plan_hash"]:
        state["previous_plan_hash"] = previous.get("plan_hash")
        state["plan_changed_at"] = _now_iso()
    return state


def _mark_segment(state: dict[str, Any], seg: SegmentRenderStep, status: str, **meta: Any) -> None:
    entry = {
        "status": status,
        "index": seg.index,
        "output_mp4": seg.output_mp4,
        "hero_out": seg.hero_out,
        "frames": seg.frames,
        "updated_at": _now_iso(),
    }
    entry.update(meta)
    if status == "complete":
        entry.setdefault("completed_at", _now_iso())
    state.setdefault("segments", {})[str(seg.index)] = entry
    state["updated_at"] = _now_iso()


def minute_render_status(plan: MinuteRenderPlan, *, state_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(state_path).expanduser().resolve() if state_path else state_path_for_plan(plan)
    state = _load_json(path)
    ph = plan_hash(plan)
    stale_state = bool(state.get("plan_hash") and state.get("plan_hash") != ph)

    seg_rows = []
    complete = 0
    output_only = 0
    missing = 0
    for seg in plan.segments:
        output_ready = file_ready(seg.output_mp4)
        hero_ready = seg.hero_out is None or file_ready(seg.hero_out)
        done = output_ready and hero_ready
        complete += int(done)
        output_only += int(output_ready and not hero_ready)
        missing += int(not output_ready)
        state_entry = state.get("segments", {}).get(str(seg.index), {})
        seg_rows.append(
            {
                "index": seg.index,
                "frames": seg.frames,
                "output_mp4": seg.output_mp4,
                "hero_out": seg.hero_out,
                "output_ready": output_ready,
                "hero_ready": hero_ready,
                "complete": done,
                "state_status": state_entry.get("status"),
            }
        )

    return {
        "state_path": str(path),
        "plan_hash": ph,
        "state_plan_hash": state.get("plan_hash"),
        "stale_state": stale_state,
        "segments_total": len(plan.segments),
        "segments_complete": complete,
        "segments_output_only": output_only,
        "segments_missing": missing,
        "full_output": plan.full_output,
        "full_output_ready": file_ready(plan.full_output),
        "concat_status": state.get("concat", {}).get("status"),
        "segments": seg_rows,
    }


def execute_minute_render_plan(
    plan: MinuteRenderPlan,
    *,
    cwd: str | Path | None = None,
    state_path: str | Path | None = None,
    skip_existing: bool = True,
    force_concat: bool = False,
) -> dict[str, Any]:
    """Run a minute render plan with per-segment resume markers."""
    path = Path(state_path).expanduser().resolve() if state_path else state_path_for_plan(plan)
    old = _load_json(path)
    ph = plan_hash(plan)
    state = old if old.get("plan_hash") == ph else _init_state(plan, previous=old)
    if not state.get("segments"):
        state.setdefault("segments", {})
    state["plan_hash"] = ph
    state["out_dir"] = plan.out_dir
    state["full_output"] = plan.full_output
    state["updated_at"] = _now_iso()
    _atomic_write_json(path, state)

    can_skip_by_file = skip_existing and not bool(state.get("previous_plan_hash"))
    workdir = str(cwd) if cwd else None

    for seg in plan.segments:
        if can_skip_by_file and file_ready(seg.output_mp4):
            if seg.extract_hero_shell and not file_ready(seg.hero_out):
                subprocess.run(seg.extract_hero_shell, shell=True, check=True, cwd=workdir)
            if _segment_file_complete(seg):
                print(f"==> segment {seg.index} already complete; skipping", flush=True)
                _mark_segment(state, seg, "complete", skipped_existing=True)
                _atomic_write_json(path, state)
                continue

        print(f"==> segment {seg.index} ({seg.frames} frames)", flush=True)
        _mark_segment(state, seg, "running", started_at=_now_iso())
        _atomic_write_json(path, state)
        subprocess.run(seg.infer_shell, shell=True, check=True, cwd=workdir)
        if not file_ready(seg.output_mp4):
            _mark_segment(state, seg, "failed", error="missing output after inference")
            _atomic_write_json(path, state)
            raise RuntimeError(f"segment {seg.index} did not produce {seg.output_mp4}")
        if seg.extract_hero_shell:
            subprocess.run(seg.extract_hero_shell, shell=True, check=True, cwd=workdir)
            if not file_ready(seg.hero_out):
                _mark_segment(state, seg, "failed", error="missing hero after extraction")
                _atomic_write_json(path, state)
                raise RuntimeError(f"segment {seg.index} did not produce hero {seg.hero_out}")
        _mark_segment(state, seg, "complete")
        _atomic_write_json(path, state)

    if skip_existing and not force_concat and file_ready(plan.full_output):
        print(f"==> concat already complete; skipping {plan.full_output}", flush=True)
        state["concat"] = {"status": "complete", "output_mp4": plan.full_output, "skipped_existing": True, "updated_at": _now_iso()}
        _atomic_write_json(path, state)
    else:
        print("==> concat", flush=True)
        state["concat"] = {"status": "running", "output_mp4": plan.full_output, "started_at": _now_iso()}
        _atomic_write_json(path, state)
        subprocess.run(plan.concat_shell, shell=True, check=True, cwd=workdir)
        if not file_ready(plan.full_output):
            state["concat"] = {"status": "failed", "output_mp4": plan.full_output, "error": "missing concat output", "updated_at": _now_iso()}
            _atomic_write_json(path, state)
            raise RuntimeError(f"concat did not produce {plan.full_output}")
        state["concat"] = {"status": "complete", "output_mp4": plan.full_output, "completed_at": _now_iso()}
        _atomic_write_json(path, state)

    return minute_render_status(plan, state_path=path)
