"""Optional live ALFWorld adapter (requires `alfworld` package + data)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ltx_trainer.role_agent.types import TaskSpec


def alfworld_importable() -> bool:
    try:
        import alfworld  # noqa: F401

        return True
    except ImportError:
        return False


def alfworld_data_configured() -> bool:
    data_root = os.environ.get("ALFWORLD_DATA")
    if data_root and os.path.isdir(data_root):
        return True
    default = os.path.expanduser("~/.cache/alfworld")
    return os.path.isdir(default)


def probe_alfworld() -> dict[str, Any]:
    return {
        "package": alfworld_importable(),
        "data": alfworld_data_configured(),
        "ready": alfworld_importable() and alfworld_data_configured(),
        "install": "pip install alfworld && alfworld-download",
        "env_var": "ALFWORLD_DATA",
    }


@dataclass
class AlfWorldLiveEnv:
    """Thin wrapper around AlfredTWEnv when alfworld is installed."""

    task: TaskSpec
    state: str = ""
    done: bool = False
    success: bool = False
    backend: str = "alfworld"
    _env: Any = None
    _admissible: list[str] = field(default_factory=list)
    _step: int = 0
    _max_steps: int = 50

    @classmethod
    def available(cls) -> bool:
        return probe_alfworld()["ready"]

    @classmethod
    def create(cls, *, split: str = "eval_out_of_distribution", max_steps: int = 50) -> AlfWorldLiveEnv:
        if not alfworld_importable():
            raise RuntimeError("alfworld not installed; pip install alfworld")
        from alfworld.agents.environment import get_environment

        env_type = os.environ.get("GOPEX_ALFWORLD_ENV_TYPE", "AlfredTWEnv")
        config = {
            "env": {"type": env_type},
            "dataset": {
                "data_path": os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
                "eval_id_data_path": os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
            },
        }
        env = get_environment(env_type)(config, train_eval=split)
        env = env.init_env(batch_size=1)
        task = TaskSpec(
            task_id=f"alfworld_live_{split}",
            domain="alfworld",
            prompt="ALFWorld embodied task (live)",
            objective=split,
        )
        return cls(task=task, _env=env, _max_steps=max_steps)

    def reset(self) -> str:
        obs, _info = self._env.reset()
        self.state = _obs_text(obs)
        self._admissible = _admissible_from_info(_info)
        self.done = False
        self.success = False
        self._step = 0
        return self.state

    def valid_actions(self) -> list[str]:
        return list(self._admissible)

    def step_action(self, action: str) -> tuple[str, float, bool]:
        if self.done:
            return self.state, 0.0, True
        self._step += 1
        obs, scores, dones, infos = self._env.step([action])
        self.state = _obs_text(obs)
        self._admissible = _admissible_from_info(infos)
        reward = float(scores[0]) if scores else 0.0
        self.done = bool(dones[0]) if dones is not None else False
        self.success = self.done and reward > 0
        if self._step >= self._max_steps and not self.done:
            self.done = True
        return self.state, reward, self.done


def _obs_text(obs: Any) -> str:
    if isinstance(obs, list) and obs:
        obs = obs[0]
    return str(obs or "").strip()


def _admissible_from_info(info: Any) -> list[str]:
    if isinstance(info, list) and info:
        info = info[0]
    if isinstance(info, dict):
        cmds = info.get("admissible_commands") or info.get("admissible") or []
        return [str(c) for c in cmds]
    return []
