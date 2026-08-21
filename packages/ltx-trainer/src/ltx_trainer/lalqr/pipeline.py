"""Framework card, demos, and benchmark manifest (arXiv:2606.04775)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lalqr.config import LalqrConfig
from ltx_trainer.lalqr.contrastive_prompts import list_categories, pair_for
from ltx_trainer.lalqr.infer_bridge import ltx_integration_notes
from ltx_trainer.lalqr.metrics import (
    average_violation,
    projection_calibrated_setpoint,
    table1_t2vsafetybench,
    table1_vbench_subject,
    table2_safesora,
)
from ltx_trainer.lalqr.steering import run_lalqr_smoke


def framework_card(cfg: LalqrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LalqrConfig()
    return {
        "name": "LA-LQR",
        "paper": cfg.paper_arxiv,
        "task": "Reduced-order linear optimal control for T2V activation steering",
        "components": [
            "Contrastive reduced-order subspace (randomized SVD)",
            "Latent linear feature setpoints (LLFS)",
            "LTV LQR Riccati feedback on text embeddings",
            "Projection-calibrated raw/latent setpoint bounds (Lemma 4.1)",
        ],
        "intervention_default": cfg.intervention,
        "models": list(cfg.models),
        "benchmarks": list(cfg.benchmarks),
        "latent_rank": cfg.latent_rank,
        "contrastive_pairs": cfg.contrastive_pairs,
        "ltx_bridge": ltx_integration_notes(cfg),
    }


def paper_limitations() -> list[str]:
    return [
        "Full DiT activation-space JVP control requires white-box per-layer hooks and large basis storage.",
        "Proxy LTX bridge steers pooled text connector embeddings, not per-layer video-token activations.",
        "Synthetic local dynamics in this stub stand in for fitted Jacobians across prompts.",
        "Category routing uses keyword heuristics unless GOPEX_LALQR_CATEGORY is set.",
        "Pair with prompt filtering / output moderation — LA-LQR is not a standalone safety guarantee.",
    ]


def benchmark_manifest(cfg: LalqrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LalqrConfig()
    t1 = table1_t2vsafetybench()
    return {
        "primary": "T2VSafetyBench (Wan2.1-T2V-14B + LightX2V)",
        "secondary": "SafeSora (HunyuanVideo-1.5 + LightX2V)",
        "metrics": ["Violation Rate", "VBench Subject Consistency", "CAPS"],
        "evaluator": "GPT-4o per-video binary unsafe label (T2VSafetyBench protocol)",
        "categories_wan": list(cfg.wan_categories),
        "categories_hunyuan": list(cfg.hunyuan_categories),
        "table1_la_lqr_avg_violation": average_violation(t1, method="LA-LQR", benchmark="T2VSafetyBench"),
        "table2_la_lqr_avg_violation": average_violation(table2_safesora(), method="LA-LQR", benchmark="SafeSora"),
    }


def evaluation_demo(*, cfg: LalqrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LalqrConfig()
    smoke = run_lalqr_smoke(cfg, category="Pornography")
    smoke_gore = run_lalqr_smoke(cfg, category="Gore", seed=7)
    pos, neg = pair_for("Pornography")
    rho = float(smoke["rho"])
    return {
        "smoke_pornography": smoke,
        "smoke_gore": smoke_gore,
        "contrastive_example": {"positive": pos[:80] + "…", "negative": neg[:80] + "…"},
        "projection_setpoint_demo": projection_calibrated_setpoint(1.0, rho=rho),
        "table1_la_lqr_avg_violation": benchmark_manifest(cfg)["table1_la_lqr_avg_violation"],
        "table2_la_lqr_avg_violation": benchmark_manifest(cfg)["table2_la_lqr_avg_violation"],
        "categories": list_categories(),
        "vbench_la_lqr_gore": next(
            r["value"] for r in table1_vbench_subject() if r["category"] == "Gore"
        ),
    }


def table1_summary() -> list[dict[str, Any]]:
    return table1_t2vsafetybench()


def table2_summary() -> list[dict[str, Any]]:
    return table2_safesora()
