"""Bridge Role-Agent config to VeRL / upstream AMAP-ML/roleagent launch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.upstream import upstream_paths, verl_launch_plan


def verl_config_dict(cfg: RoleAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RoleAgentConfig()
    model = os.environ.get("GOPEX_ROLE_AGENT_MODEL", cfg.backbone_default)
    return {
        "algorithm": "role_agent",
        "model": model,
        "learning_rate": cfg.learning_rate,
        "train_batch_size": {"alfworld": 16, "webshop": 16, "search_qa": 256},
        "group_size": cfg.group_size,
        "gamma": cfg.gamma,
        "kl_coefficient": cfg.kl_coefficient,
        "clip_ratio_low": cfg.clip_ratio_low,
        "clip_ratio_high": cfg.clip_ratio_high,
        "state_similarity_threshold": cfg.state_similarity_threshold,
        "advantage_alpha": cfg.advantage_alpha,
        "rollout_temperature": cfg.rollout_temperature,
        "reflection_temperature": cfg.reflection_temperature,
        "total_epochs": cfg.total_epochs,
        "t_max": dict(cfg.t_max),
        "prediction_horizon_frac": cfg.prediction_horizon_frac,
        "wia": {"enabled": cfg.enable_wia, "horizon_frac": cfg.prediction_horizon_frac},
        "aiw": {"enabled": cfg.enable_aiw, "failure_memory": True, "llm_retrieval": True},
        "search_retriever": cfg.search_retriever,
        "gigpo": {"state_grouping": True, "similarity_threshold": cfg.state_similarity_threshold},
    }


def write_verl_config(path: Path | None = None, cfg: RoleAgentConfig | None = None) -> Path:
    cfg = cfg or RoleAgentConfig()
    paths = upstream_paths()
    out = path or (paths["clone_dir"] / "gopex_role_agent_verl.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = verl_config_dict(cfg)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def launch_commands(cfg: RoleAgentConfig | None = None, *, gpus: int = 8) -> list[str]:
    cfg = cfg or RoleAgentConfig()
    paths = upstream_paths()
    clone = paths["clone_dir"]
    config_path = clone / "gopex_role_agent_verl.json"
    plan = verl_launch_plan(cfg, gpus=gpus)
    return [
        f"mkdir -p {clone.parent}",
        f"git clone --depth 1 {cfg.github}.git {clone} || true",
        f"cd {clone} && pip install -e .",
        "pip install verl alfworld webshop faiss-cpu sentence-transformers",
        f"python -c \"from ltx_trainer.role_agent.verl_bridge import write_verl_config; write_verl_config()\"",
        f"# Config written to {config_path}",
        f"export GOPEX_ROLE_AGENT_MODEL={plan['backbone']}",
        f"export CUDA_VISIBLE_DEVICES=0-{gpus - 1}",
        f"cd {clone} && bash scripts/train.sh  # upstream entry; see README",
    ]


def bridge_status() -> dict[str, Any]:
    cfg = RoleAgentConfig()
    paths = upstream_paths()
    config_path = paths["clone_dir"] / "gopex_role_agent_verl.json"
    return {
        "verl_plan": verl_launch_plan(cfg),
        "config_path": str(config_path),
        "config_exists": config_path.is_file(),
        "config_preview": verl_config_dict(cfg),
        "launch_commands": launch_commands(cfg),
    }
