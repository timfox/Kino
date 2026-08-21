"""Evaluation demo and smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.step37_flash.api import build_chat_request
from ltx_trainer.step37_flash.architecture import architecture_card, reasoning_level_tradeoff
from ltx_trainer.step37_flash.benchmarks import benchmarks_bundle
from ltx_trainer.step37_flash.config import Step37FlashConfig
from ltx_trainer.step37_flash.core import active_param_fraction, estimate_cost_usd, throughput_card
from ltx_trainer.step37_flash.deployment import deployment_card, vllm_serve_command
from ltx_trainer.step37_flash.ltx_plan import ltx_caption_plan


def evaluation_demo(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Step37FlashConfig()
    cost = estimate_cost_usd(input_tokens=50_000, output_tokens=2_000, cache_hit_ratio=0.3)
    return {
        "architecture": architecture_card(),
        "throughput": throughput_card(),
        "active_fraction": active_param_fraction(),
        "benchmarks": benchmarks_bundle(),
        "reasoning_medium": reasoning_level_tradeoff("medium"),
        "sample_chat_request": build_chat_request("Introduce StepFun AI capabilities."),
        "sample_vision_request": build_chat_request(
            "What is in this picture?",
            image_url="https://example.com/photo.jpg",
            reasoning_level="low",
        ),
        "cost_estimate_50k_in_2k_out": cost,
        "vllm_fp8_cmd_preview": vllm_serve_command(precision="fp8")[:120] + "...",
        "deployment_vllm": deployment_card("vllm"),
        "ltx_plan": ltx_caption_plan(),
        "github": cfg.github_repo,
    }


def evaluation_smoke(cfg: Step37FlashConfig | None = None) -> dict[str, Any]:
    cfg = cfg or Step37FlashConfig()
    ev = evaluation_demo(cfg)
    arch = ev["architecture"]
    assert arch["active_params_per_token_b"] == 11.0
    assert arch["max_context_tokens"] >= 200_000
    assert ev["benchmarks"]["agent_tools"]["ClawEval_1_1"] > 60.0
    assert ev["cost_estimate_50k_in_2k_out"]["usd"] > 0
    return {
        "package": "ltx_trainer.step37_flash",
        "status": "smoke_ok",
        "model": cfg.api_model_id,
        "ok": True,
        "active_fraction": ev["active_fraction"],
        "swe_bench_pro": ev["benchmarks"]["coding_professional"]["SWE_Bench_PRO"],
    }
