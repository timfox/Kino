"""HarnessAPI skill-folder bridge: advertise skills from local paths (Sec. 3.1, 5.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ltx_trainer.swarmharness.config import ResourceVector, SwarmNode
from ltx_trainer.swarmharness.identity import NodeIdentity


@dataclass
class SkillManifest:
    """Minimal skill descriptor read from a HarnessAPI-style folder."""

    name: str
    path: Path
    vram_gb: float = 0.0
    description: str = ""


def _read_manifest(skill_dir: Path) -> SkillManifest | None:
    meta = skill_dir / "skill.json"
    if meta.is_file():
        data = json.loads(meta.read_text(encoding="utf-8"))
        return SkillManifest(
            name=str(data.get("name", skill_dir.name)),
            path=skill_dir,
            vram_gb=float(data.get("vram_gb", 0.0)),
            description=str(data.get("description", "")),
        )
    if (skill_dir / "SKILL.md").is_file() or (skill_dir / "handler.py").is_file():
        return SkillManifest(name=skill_dir.name, path=skill_dir)
    return None


def discover_skills(skills_root: Path) -> list[SkillManifest]:
    """
    Scan ``skills_root`` for HarnessAPI skill folders.

    Layout::
        skills_root/
          inference/skill.json
          summarize/handler.py
    """
    root = skills_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"skills root not found: {root}")
    out: list[SkillManifest] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        manifest = _read_manifest(child)
        if manifest:
            out.append(manifest)
    return out


def node_from_skill_folder(
    skills_root: Path,
    *,
    identity: NodeIdentity | None = None,
    vram_gb: float | None = None,
) -> SwarmNode:
    """Build a SwarmNode advertisement from a local skill directory tree."""
    ident = identity or NodeIdentity.generate()
    skills = discover_skills(skills_root)
    max_vram = max((s.vram_gb for s in skills), default=0.0)
    return SwarmNode(
        node_id=ident.node_id,
        skills={s.name for s in skills},
        resources=ResourceVector(vram_gb=vram_gb if vram_gb is not None else max_vram),
        public_key_hex=ident.public_key_hex,
        genesis_locked=True,
    )


def harness_bridge_card(skills_root: Path) -> dict[str, Any]:
    skills = discover_skills(skills_root)
    return {
        "skills_root": str(skills_root.resolve()),
        "skills": [{"name": s.name, "vram_gb": s.vram_gb, "path": str(s.path)} for s in skills],
        "count": len(skills),
    }
