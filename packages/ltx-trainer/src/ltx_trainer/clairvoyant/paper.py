"""Framework card, knowledge blob, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clairvoyant.benchmarks import benchmarks_bundle
from ltx_trainer.clairvoyant.config import PAPER_ARXIV, PAPER_TITLE, PAPER_URL, UPSTREAM_REPO
from ltx_trainer.clairvoyant.constants import (
    MEDIUM_MAX_TOKENS,
    SHORT_MAX_TOKENS,
    STARVATION_TAU_MULTIPLIER,
)
from ltx_trainer.clairvoyant.datasets import datasets_card
from ltx_trainer.clairvoyant.features import feature_names


def framework_card() -> dict[str, Any]:
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "upstream": UPSTREAM_REPO,
        "architecture": [
            "feature_extractor_19_lexical",
            "onnx_xgboost_3class",
            "sjf_min_heap_p_long",
            "starvation_tau_3x_mu_short",
        ],
        "target_backends": ["Ollama", "llama.cpp", "Jan"],
        "not_for": ["vLLM", "Orca", "TGI continuous batching"],
        "class_boundaries_tokens": {
            "short": f"< {SHORT_MAX_TOKENS}",
            "medium": f"[{SHORT_MAX_TOKENS}, {MEDIUM_MAX_TOKENS})",
            "long": f">= {MEDIUM_MAX_TOKENS}",
        },
        "starvation_tau": f"{STARVATION_TAU_MULTIPLIER} × μ_short",
        "n_features": len(feature_names()),
    }


def evaluation_demo() -> dict[str, Any]:
    return {
        "package": "clairvoyant",
        "framework": framework_card(),
        "benchmarks": benchmarks_bundle(),
        "datasets": datasets_card(),
    }


def knowledge_blob() -> dict[str, Any]:
    bench = benchmarks_bundle()
    return {
        "title": PAPER_TITLE,
        "arxiv": PAPER_ARXIV,
        "url": PAPER_URL,
        "upstream": UPSTREAM_REPO,
        "summary": (
            "Drop-in OpenAI-compatible sidecar for serial LLM backends (Ollama, llama.cpp). "
            "Predicts output length from 19 lexical features via ONNX XGBoost (~0.029 ms), "
            "dispatches requests in non-preemptive SJF order on P(Long) with starvation timeout "
            "τ = 3×μ_short to mitigate Layer-1 head-of-line blocking without extra KV-cache VRAM."
        ),
        "predictor_latency_ms": bench["predictor_latency_ms"]["sharegpt_onnx"],
        "ranking_accuracy_id_pct": "62–96",
        "burst_short_p50_reduction_pct": "70–76",
        "steady_short_p50_reduction_pct_rho_074": 17,
        "deployment_rho_band": bench["deployment_rho_band"],
        "instruction_datasets_degenerate": datasets_card()["instruction_datasets_degenerate"],
    }
