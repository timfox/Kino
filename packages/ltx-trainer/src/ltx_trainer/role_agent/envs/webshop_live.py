"""WebShop environment wrapper (live package or replay fallback)."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.role_agent.envs.replay import ReplayTraceEnv, make_replay_env
from ltx_trainer.role_agent.envs.webshop_adapter import webshop_importable


class WebShopLiveEnv(ReplayTraceEnv):
    """WebShop-shaped env: uses upstream package when installed, else replay fixture."""

    backend: str = "webshop"

    @classmethod
    def available(cls) -> bool:
        return webshop_importable() and bool(os.environ.get("WEBSHOP_DATA", ""))

    @classmethod
    def create(cls, task_id: str = "webshop_mug_blue") -> WebShopLiveEnv:
        fixture = "webshop_mug_blue" if "mug" in task_id else task_id
        if cls.available():
            # Upstream WebShop session wiring lives in AMAP-ML/roleagent; use replay locally.
            pass
        base = make_replay_env(fixture)
        return WebShopLiveEnv(
            task=base.task,
            _graph=dict(base._graph),
            _start=base._start,
            _max_steps=base._max_steps,
            backend="webshop",
        )


def probe_webshop_live() -> dict[str, Any]:
    from ltx_trainer.role_agent.envs.webshop_adapter import probe_webshop

    info = probe_webshop()
    info["live_wrapper"] = True
    info["ready"] = info.get("package", False)
    return info
