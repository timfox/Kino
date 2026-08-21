"""Skill compiler G_φ: text → LoRA (§3.1–3.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.lora import LatentSkillAdapter, compile_skill_lora


def adapter_digest(text: str) -> str:
    return compile_skill_lora(text, "tmp").text_digest


@dataclass
class SkillCompiler:
    """Transformer hypernetwork stub with adapter cache."""

    cfg: LatentSkillConfig = field(default_factory=LatentSkillConfig)
    cache: dict[str, LatentSkillAdapter] = field(default_factory=dict)

    def compile(self, skill_text: str, skill_id: str, *, use_cache: bool = True) -> LatentSkillAdapter:
        if use_cache and skill_id in self.cache:
            return self.cache[skill_id]
        adapter = compile_skill_lora(skill_text, skill_id, self.cfg)
        if use_cache:
            self.cache[skill_id] = adapter
        return adapter

    def pretrain_step(self, skill_doc: str, *, task: str = "reconstruction") -> dict[str, Any]:
        """Document-level L_pre stub (Eq. 4): compiler-only update signal."""
        adapter = self.compile(skill_doc, skill_id=f"pre_{adapter_digest(skill_doc)}", use_cache=False)
        target_len = len(skill_doc.split())
        pred_len = int(target_len * (0.85 + 0.1 * np.tanh(adapter.frobenius_mean())))
        loss = abs(pred_len - target_len) / max(target_len, 1)
        return {"task": task, "loss": float(loss), "n_updates": len(adapter.updates)}

    def sft_step(self, skill_doc: str, trajectory_steps: int) -> dict[str, Any]:
        """Trajectory L_sft stub (Eq. 5): shared adapter across steps."""
        adapter = self.compile(skill_doc, skill_id=f"sft_{adapter_digest(skill_doc)}", use_cache=False)
        consistency = 1.0 - 0.05 * np.log1p(trajectory_steps) / np.log1p(50)
        return {
            "trajectory_steps": trajectory_steps,
            "shared_adapter": True,
            "consistency_score": float(max(consistency, 0.5)),
            "n_updates": len(adapter.updates),
        }
