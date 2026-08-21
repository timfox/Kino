"""Export compiled skill adapters for vLLM / PEFT (stub manifest)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.latentskill.compiler import SkillCompiler
from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.skills import SKILL_DOCUMENTS


def default_export_root() -> Path:
    work = os.environ.get("WORK_ROOT", os.path.expanduser("~/gopex-work"))
    return Path(work) / "adapters" / "latentskill"


def export_adapter_manifest(
    skill_names: list[str] | None = None,
    *,
    cfg: LatentSkillConfig | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Write manifest.json listing stub LoRA digests (no real safetensors in CPU stub)."""
    cfg = cfg or LatentSkillConfig()
    out_dir = out_dir or default_export_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler = SkillCompiler(cfg)
    names = skill_names or list(SKILL_DOCUMENTS.keys())
    entries: list[dict[str, Any]] = []
    for name in names:
        sid = name.replace(" ", "_").lower()
        adapter = compiler.compile(SKILL_DOCUMENTS[name], sid)
        entries.append(
            {
                "skill_id": sid,
                "skill_name": name,
                "text_digest": adapter.text_digest,
                "n_updates": len(adapter.updates),
                "modules": list({u.module for u in adapter.updates}),
                "layers": sorted({u.layer for u in adapter.updates}),
                "safetensors": str(out_dir / f"{sid}.safetensors"),
                "status": "manifest_only_stub",
            }
        )
    manifest = {
        "backbone": cfg.backbone,
        "lora_rank": cfg.lora_rank,
        "preferred_modules": list(cfg.preferred_modules),
        "default_alpha": cfg.default_injection_alpha,
        "skills": entries,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"export_dir": str(out_dir), "manifest": str(path), "n_skills": len(entries), "skills": entries}
