"""Search-QA environment probe (E5 / search-augmented QA upstream)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.envs.search_qa_env import SearchQAEnv, list_search_qa_task_ids


def search_qa_importable() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def probe_search_qa() -> dict[str, Any]:
    return {
        "package": search_qa_importable(),
        "local_env": True,
        "tasks": list_search_qa_task_ids(),
        "ready": True,
        "note": "SearchQAEnv corpus available locally; full E5+VeRL via upstream roleagent",
        "install": "pip install faiss-cpu sentence-transformers; follow AMAP-ML/roleagent for scale training",
        "env_class": SearchQAEnv.__name__,
    }
