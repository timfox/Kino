"""VCap e1→e2 self-improvement helpers (witness pool backup + reference swap)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def witness_backup_path(manifest_dir: Path, epoch: str) -> Path:
    return manifest_dir / f"dataset.{epoch}.json"


def backup_witness_pool(manifest_dir: str | Path, *, epoch: str = "e1") -> dict[str, Any]:
    """Copy ``dataset.json`` → ``dataset.{epoch}.json`` (idempotent if already newer)."""
    root = Path(manifest_dir).expanduser().resolve()
    src = root / "dataset.json"
    dst = witness_backup_path(root, epoch)
    if not src.is_file():
        return {"ok": False, "error": f"missing {src}", "backup": None}
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return {"ok": True, "backup": str(dst), "skipped": True, "reason": "backup not older than source"}
    shutil.copy2(src, dst)
    rows = json.loads(src.read_text(encoding="utf-8"))
    n_rows = len(rows) if isinstance(rows, list) else 0
    return {"ok": True, "backup": str(dst), "skipped": False, "rows": n_rows}


def batch_backup_projects(
    datasets_root: str | Path,
    projects: list[str],
    *,
    epoch: str = "e1",
) -> dict[str, Any]:
    """Backup witness pools for ``<root>/<project>/ltx_manifest``."""
    base = Path(datasets_root).expanduser().resolve()
    out: dict[str, Any] = {"epoch": epoch, "projects": {}}
    for name in projects:
        name = name.strip()
        if not name:
            continue
        man = base / name / "ltx_manifest"
        if not (man / "dataset.json").is_file():
            man = base / name
        out["projects"][name] = backup_witness_pool(man, epoch=epoch)
    return out


def recaption_command(
    project: str,
    *,
    datasets_root: str | Path,
    repo: str | Path,
) -> list[str]:
    """Shell argv to resume split+caption for one project (VCap dense prompts via env)."""
    root = Path(datasets_root).expanduser().resolve() / project
    manifest = root / "ltx_manifest"
    out_dir = manifest if manifest.is_dir() else root
    script = Path(repo).expanduser().resolve() / "tools" / "dataset_split_and_caption.py"
    py = Path(repo) / ".venv/bin/python"
    if not py.is_file():
        py = Path("python")
    return [
        f"GOPEX_VCAP_CAPTION=1 GOPEX_VCAP_EPOCH=e2",
        str(py),
        str(script),
        "--dataset-root",
        str(root),
        "--output-dir",
        str(out_dir),
        "# default: resumes uncaptioned clips only (omit --no-resume)",
    ]


def e2_workflow_card() -> dict[str, Any]:
    """Steps for weak-to-strong reference regeneration (paper Fig. 2c)."""
    return {
        "epoch_e1": [
            "Train or caption with weak reference pool (backbone captions).",
            "backup_witness_pool(manifest, epoch='e1') after e1 captions stable.",
            "GOPEX_VCAP_CAPTION=1 native prep + GRPO (external) on e1 witnesses.",
        ],
        "epoch_e2": [
            "Regenerate references with improved policy (re-run dataset_split_and_caption).",
            "backup stays at dataset.e1.json; set GOPEX_VCAP_REFERENCE_DATASET to it.",
            "GOPEX_VCAP_EPOCH=e2; qa + merge with sharpened witness size m.",
        ],
        "commands": [
            "./scripts/kino-vcap-evolve.sh backup-e1",
            "./scripts/kino-vcap-evolve.sh qa-e2",
            "./scripts/kino-vcap-evolve.sh print-recaption",
        ],
    }
