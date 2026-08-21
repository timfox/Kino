"""Framework card, paper tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.anthropocam.config import (
    LATENCY_TARGET_S,
    MOBILE_RESOLUTION,
    OPTIMAL_BATCH_SIZE,
    OPTIMAL_EPOCHS,
    OPTIMAL_STYLE_WEIGHT,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    AnthropoCamConfig,
)
from ltx_trainer.anthropocam.training import (
    batch_size_table,
    epoch_convergence_table,
    forward_smoke,
    resolution_latency_table,
    style_weight_sensitivity,
)


def framework_card(cfg: AnthropoCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AnthropoCamConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "url": cfg.paper_url,
        "system": "AnthropoCam",
        "domain": "Anthropocene landscape NST (toxic sublime)",
        "backbone": cfg.backbone,
        "inference": cfg.inference_mode,
        "mobile_stack": {
            "frontend": cfg.stack_frontend,
            "backend": cfg.stack_backend,
            "latency_target_s": list(cfg.latency_target_s),
            "resolution": [cfg.mobile_width, cfg.mobile_height],
        },
        "loss": "alpha*content + beta*style(Gram) + gamma*TV",
        "optimal_style_weight": OPTIMAL_STYLE_WEIGHT,
        "optimal_epochs": OPTIMAL_EPOCHS,
        "optimal_batch_size": OPTIMAL_BATCH_SIZE,
    }


def knowledge_blob() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "summary": (
            "AnthropoCam is a mobile feed-forward NST pipeline for Anthropocene field capture. "
            "VGG-16 Gram statistics amplify industrial textures while conv3_3 content loss preserves "
            "semantic legibility. Optimal w_s=5, 10 epochs, batch 8, 1280×2276 mobile resolution, 3–5s latency."
        ),
        "latency_target_s": list(LATENCY_TARGET_S),
        "mobile_resolution": list(MOBILE_RESOLUTION),
        "style_weight_optimal": OPTIMAL_STYLE_WEIGHT,
    }


def evaluation_demo(cfg: AnthropoCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AnthropoCamConfig()
    return {
        "framework": framework_card(cfg),
        "knowledge": knowledge_blob(),
        "style_weight_sensitivity": style_weight_sensitivity(),
        "epoch_convergence": epoch_convergence_table(),
        "batch_size_stability": batch_size_table(),
        "resolution_latency": resolution_latency_table(),
        "forward": forward_smoke(cfg),
    }
