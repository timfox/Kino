"""Map Diffusion LM agent stack to GOPEX packages."""

from __future__ import annotations

from typing import Any


def gopex_integration_map() -> dict[str, str]:
    return {
        "gopex_diffusion_lm": "Hybrid turn planner + messenger worker",
        "gopex_agent/local_model_runtime": "AR backbone routing (/model qwen|gemma4|…)",
        "tools/vllm_chat_qt.py": "Parameters → Diffusion LM hybrid mode",
        "tools/gopex_diffusion_lm_worker.py": "Off-UI-thread hybrid turns",
        "gopex_tot": "Classical search deliberation (orthogonal to dLLM aux)",
        "gopex_core": "Contrastive insight memory after turns",
        "ltx_trainer/dlmasr": "Parallel masked decode thresholds for speech",
        "ltx_trainer/constraint_tax": "Schema guard for tool JSON (delay constrain late)",
        "documents/DIFFUSION_LM.md": "Architecture + env vars",
    }


def messenger_env_vars() -> list[dict[str, str]]:
    return [
        {"name": "GOPEX_DLLM_MODE", "default": "0", "purpose": "Enable hybrid worker in messenger"},
        {"name": "GOPEX_DLLM_PROFILE", "default": "hybrid_auto", "purpose": "hybrid_auto|diffuagent|p_react"},
        {"name": "GOPEX_DLLM_BASE_URL", "default": "(active AR URL)", "purpose": "Optional dedicated dLLM endpoint"},
        {"name": "GOPEX_DLLM_DECODE", "default": "dynamic_threshold", "purpose": "Parallel commit strategy"},
        {"name": "GOPEX_DLLM_MAX_AUX", "default": "3", "purpose": "Cap auxiliary role LLM calls"},
    ]


def integration_bundle() -> dict[str, Any]:
    return {
        "gopex_map": gopex_integration_map(),
        "env": messenger_env_vars(),
    }
