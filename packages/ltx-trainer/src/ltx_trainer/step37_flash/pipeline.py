"""Framework and knowledge exports."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.architecture import architecture_card
from ltx_trainer.step37_flash.benchmarks import benchmarks_bundle
from ltx_trainer.step37_flash.config import Step37FlashConfig
from ltx_trainer.step37_flash.constants import AGENT_PLATFORMS, AVAILABILITY_CHANNELS
from ltx_trainer.step37_flash.deployment import deployment_card
from ltx_trainer.step37_flash.ltx_plan import ltx_caption_plan


def framework_card(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Step37FlashConfig()
    return {
        "name": cfg.name,
        "title": cfg.title,
        "team": cfg.team,
        "website": cfg.website,
        "github_repo": cfg.github_repo,
        "license": cfg.license,
        "api_model_id": cfg.api_model_id,
        "ltx_hook": cfg.ltx_hook,
        "architecture": architecture_card(),
        "benchmarks": benchmarks_bundle(),
        "availability": list(AVAILABILITY_CHANNELS),
        "agent_platforms": list(AGENT_PLATFORMS),
        "deployment_backends": ["vllm", "sglang", "transformers", "llama_cpp", "nim"],
    }


def knowledge_card(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Step37FlashConfig()
    return {
        "framework": framework_card(cfg),
        "ltx_integration": ltx_caption_plan(),
        "deploy_vllm": deployment_card("vllm"),
        "deploy_sglang": deployment_card("sglang"),
    }


def evaluation_smoke(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.step37_flash.evaluation import evaluation_smoke as _smoke

    return _smoke(cfg)


def evaluation_demo(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.step37_flash.evaluation import evaluation_demo as _demo

    return _demo(cfg)
