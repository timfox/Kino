"""Upstream VeRL / roleagent training plan."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.config import RoleAgentConfig


def run_plan(cfg: RoleAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RoleAgentConfig()
    return {
        "github": cfg.github,
        "framework": "VeRL",
        "gpus": "8× NVIDIA H20 (paper)",
        "backbones": ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
        "domains": {
            "alfworld": {"t_max": cfg.t_max["alfworld"], "H": cfg.prediction_horizon("alfworld")},
            "webshop": {"t_max": cfg.t_max["webshop"], "H": cfg.prediction_horizon("webshop")},
            "search_qa": {"t_max": cfg.t_max["search_qa"], "H": cfg.prediction_horizon("search_qa")},
        },
        "gopex_stub": "CPU toy Algorithm 1 (WIA+AIW+GRPO) + table anchors; scale-out via upstream repo",
        "gopex_train_toy": "./scripts/gopex-role-agent.sh train-toy",
        "gopex_llm_probe": "./scripts/gopex-role-agent.sh llm-probe",
        "gopex_llm_rollout": "./scripts/gopex-role-agent.sh llm-rollout",
        "vllm_env": {
            "GOPEX_ROLE_AGENT_ENV": "replay",
            "GOPEX_ROLE_AGENT_LLM": "1",
            "GOPEX_ROLE_AGENT_LLM_ROLLOUT": "1",
            "GOPEX_ROLE_AGENT_WIA_LLM": "1",
            "GOPEX_ROLE_AGENT_AIW_LLM": "1",
            "GOPEX_VLLM_QWEN_URL": "http://127.0.0.1:8002/v1",
            "GOPEX_ROLE_AGENT_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        },
        "steps": [
            "Clone AMAP-ML/roleagent and install VeRL dependencies",
            "Prepare ALFWorld / WebShop / search-QA task pools",
            "Enable WIA state prediction prompts each rollout step",
            "After failed trajectories, run AIW reflection + failure memory update",
            "Resample training batch from curriculum_resample_weights",
            "Evaluate Table 1/2 metrics vs GiGPO baseline",
        ],
    }
