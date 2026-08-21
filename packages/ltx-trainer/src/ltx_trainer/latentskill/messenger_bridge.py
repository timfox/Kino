"""vLLM messenger + chat sidecar hooks for LatentSkill LoRA mounting."""
from __future__ import annotations

import os
from typing import Any

from ltx_trainer.latentskill.config import LatentSkillConfig
from ltx_trainer.latentskill.export import default_export_root, export_adapter_manifest
from ltx_trainer.latentskill.inference import inference_notes


def messenger_env_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    """Environment variables for GOPEX vLLM messenger when using latent skills."""
    cfg = cfg or LatentSkillConfig()
    export_dir = default_export_root()
    return {
        "GOPEX_LATENTSKILL_ALPHA": str(cfg.default_injection_alpha),
        "GOPEX_LATENTSKILL_EXPORT": str(export_dir),
        "GOPEX_LATENTSKILL_MODE": "latent",  # latent | in_context
        "vllm_enable_lora": "1",
        "vllm_lora_modules": "skill_pick,skill_look,skill_clean,direct_retrieval,multi_hop_reasoning",
    }


def messenger_integration_plan(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    return {
        "chat_ui": "tools/vllm_chat_qt.py — agent presets via vllm_messenger_presets",
        "compile_tool": "latentskill_compile",
        "export_tool": "latentskill_export",
        "rollout_tool": "latentskill_rollout",
        "role_agent_stack": "./scripts/gopex-role-agent.sh env-probe",
        "env": messenger_env_plan(cfg),
        "serve_example": (
            "vllm serve Qwen/Qwen3-8B --enable-lora "
            f"--lora-modules skill_clean={default_export_root()}/clean_skill"
        ),
        "inference": inference_notes(cfg),
        "workflow": [
            "latentskill_export → manifest under $WORK_ROOT/adapters/latentskill/",
            "vLLM start with --enable-lora and per-skill modules",
            "match_alfworld_skill / match_search_qa_skill → select module per episode",
            "Set GOPEX_LATENTSKILL_ALPHA for injection strength (Table 11 peak ~0.5–0.6)",
        ],
    }


def messenger_status(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LatentSkillConfig()
    export_dir = default_export_root()
    manifest_exists = (export_dir / "manifest.json").is_file()
    return {
        "integration": messenger_integration_plan(cfg),
        "export_dir": str(export_dir),
        "manifest_exists": manifest_exists,
        "work_root": os.environ.get("WORK_ROOT", ""),
    }


def write_export_manifest(cfg: LatentSkillConfig | None = None) -> dict[str, Any]:
    return export_adapter_manifest(cfg=cfg)
