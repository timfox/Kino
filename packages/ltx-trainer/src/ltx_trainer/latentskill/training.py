"""Two-stage training stubs (§3.2–3.3, Appendix A)."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.compiler import SkillCompiler
from ltx_trainer.latentskill.config import LatentSkillConfig


def pretrain_curriculum(cfg: LatentSkillConfig | None = None, *, n_docs: int = 8) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    compiler = SkillCompiler(cfg)
    losses = []
    for i in range(n_docs):
        doc = f"Skill document {i}: apply when household task requires step-by-step procedure."
        task = "reconstruction" if i % 2 == 0 else "completion"
        out = compiler.pretrain_step(doc, task=task)
        losses.append(out["loss"])
    return {
        "stage": "pretrain",
        "docs": n_docs,
        "mean_loss": float(sum(losses) / max(len(losses), 1)),
        "loss_decreasing": losses[-1] <= losses[0] + 0.05,
        "epochs": cfg.pretrain_epochs,
        "lr": cfg.pretrain_lr,
    }


def sft_curriculum(cfg: LatentSkillConfig | None = None, *, n_traj: int = 6) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    compiler = SkillCompiler(cfg)
    scores = []
    for i in range(n_traj):
        skill = f"Clean Skill variant {i % 3}"
        out = compiler.sft_step(skill, trajectory_steps=20 + i * 3)
        scores.append(out["consistency_score"])
    return {
        "stage": "sft",
        "trajectories": n_traj,
        "mean_consistency": float(sum(scores) / max(len(scores), 1)),
        "alfworld_traj_pool": cfg.sft_alfworld_traj,
        "search_traj_pool": cfg.sft_search_traj,
        "lora_rank_stage2": cfg.lora_rank,
        "joint_lora_with_compiler": True,
    }


def training_ladder(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "stage1": {"train": "compiler_only", "freeze": ["backbone", "vae"], "lr": cfg.pretrain_lr},
        "stage2": {"train": "compiler+lora_adapters", "freeze": ["backbone"], "lr": cfg.sft_lr},
        "pretrain_demo": pretrain_curriculum(cfg),
        "sft_demo": sft_curriculum(cfg),
    }
