"""Upstream LatentSkill / SkillRL corpus probe (stub)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig


def upstream_manifest(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "github": cfg.github,
        "pretrain_documents": cfg.pretrain_docs,
        "pretrain_tokens_m": cfg.pretrain_tokens_m,
        "skillrl_library": "171K markdown skill documents (Appendix A)",
        "install_note": "Clone upstream; replace compile_skill_lora stub with trained G_phi checkpoint",
        "gopex_path": "kino/packages/ltx-trainer/src/ltx_trainer/latentskill/",
        "status": "stub_only",
    }
