"""Upstream AMAP-ML/roleagent integration helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.envs.registry import env_catalog, probe_all_envs


def upstream_paths(work_root: Path | None = None) -> dict[str, Path]:
    root = work_root or Path(os.environ.get("WORK_ROOT", Path.home() / "gopex-work"))
    base = root / "upstream" / "roleagent"
    return {
        "work_root": root,
        "clone_dir": base,
        "verl_config": base / "verl" / "config",
        "logs": root / "logs" / "roleagent",
    }


def upstream_init_commands(cfg: RoleAgentConfig | None = None) -> list[str]:
    cfg = cfg or RoleAgentConfig()
    paths = upstream_paths()
    clone = paths["clone_dir"]
    return [
        f"mkdir -p {clone.parent}",
        f"git clone --depth 1 {cfg.github}.git {clone} || true",
        f"cd {clone} && pip install -e . || true",
        "pip install verl alfworld webshop faiss-cpu || true",
        "alfworld-download || true",
        f"# Train: see verl_launch_plan() and README in {clone}",
    ]


def verl_launch_plan(cfg: RoleAgentConfig | None = None, *, gpus: int = 8) -> dict[str, Any]:
    """VeRL + Role-Agent scale-out plan (paper: 8× H20, Qwen2.5 backbones)."""
    cfg = cfg or RoleAgentConfig()
    paths = upstream_paths()
    clone = paths["clone_dir"]
    model = os.environ.get("GOPEX_ROLE_AGENT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    return {
        "framework": "VeRL",
        "repo": cfg.github,
        "clone_dir": str(clone),
        "gpus": gpus,
        "backbone": model,
        "domains": ["alfworld", "webshop", "search_qa"],
        "hyperparams": {
            "learning_rate": cfg.learning_rate,
            "group_size": cfg.group_size,
            "gamma": cfg.gamma,
            "kl_coefficient": cfg.kl_coefficient,
            "total_epochs": cfg.total_epochs,
        },
        "env_probes": probe_all_envs(),
        "env_catalog": env_catalog(),
        "launcher": [
            f"cd {clone}",
            "# Follow upstream README: enable WIA state prediction + AIW failure memory each epoch",
            f"export GOPEX_ROLE_AGENT_MODEL={model}",
            "bash scripts/train.sh  # name may differ; check cloned repo",
        ],
        "gopex_note": "Local CPU stub: GOPEX_ROLE_AGENT_ENV=replay ./scripts/gopex-role-agent.sh train-toy",
    }


def upstream_status(work_root: Path | None = None) -> dict[str, Any]:
    paths = upstream_paths(work_root)
    clone = paths["clone_dir"]
    exists = clone.is_dir() and any(clone.iterdir()) if clone.exists() else False
    git_ok = False
    if exists and shutil.which("git"):
        try:
            subprocess.run(
                ["git", "-C", str(clone), "rev-parse", "--is-inside-work-tree"],
                check=True,
                capture_output=True,
                text=True,
            )
            git_ok = True
        except subprocess.CalledProcessError:
            git_ok = False
    cfg = RoleAgentConfig()
    return {
        "clone_dir": str(clone),
        "cloned": exists,
        "git_repo": git_ok,
        "init_commands": upstream_init_commands(cfg),
        "verl_plan": verl_launch_plan(cfg),
        "env_probes": probe_all_envs(),
    }
