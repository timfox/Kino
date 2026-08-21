"""Upstream Qwen3-8B hypernetwork training run plan."""
from __future__ import annotations

from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.training import training_ladder


def run_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    ladder = training_ladder(cfg)
    return {
        "github": cfg.github,
        "backbone": cfg.backbone,
        "hardware": "8× GPU (paper); GOPEX stub is CPU-only",
        "stage1_pretrain": {
            "objective": "L_pre document reconstruction + completion (Eq. 4)",
            "data": f"{cfg.pretrain_docs} GitHub skill documents (~{cfg.pretrain_tokens_m}M tokens)",
            "epochs": cfg.pretrain_epochs,
            "batch": cfg.pretrain_batch,
            "lr": cfg.pretrain_lr,
            "trainable": ["hypernetwork G_phi"],
            "frozen": ["Qwen3-8B backbone"],
        },
        "stage2_sft": {
            "objective": "L_sft trajectory consistency (Eq. 5)",
            "alfworld_traces": cfg.sft_alfworld_traj,
            "search_traces": cfg.sft_search_traj,
            "epochs": cfg.sft_epochs,
            "batch": cfg.sft_batch,
            "lr": cfg.sft_lr,
            "trainable": ["G_phi", "generated LoRA adapters"],
        },
        "inference": {
            "compile_cache": "C[k] = G_phi(s_k) once per skill",
            "injection": f"attn_o + mlp_down preferred; α≈{cfg.default_injection_alpha}",
            "composition": "parameter-space sum for aligned components (Eq. 6)",
        },
        "gopex_stub_ladder": ladder,
        "gopex_smoke": "./scripts/gopex-latentskill.sh smoke",
        "gopex_export": "./scripts/gopex-latentskill.sh export",
        "steps": [
            "Accept SkillRL / LatentSkill data license",
            "Stage-1 pretrain G_phi on skill document corpus",
            "Stage-2 SFT on ALFWorld + Search-QA teacher trajectories",
            "Export per-skill LoRA safetensors",
            "Serve with vLLM --enable-lora; match skill by task type",
            "Evaluate Tables 1–2 vs in-context and RAG baselines",
        ],
    }
