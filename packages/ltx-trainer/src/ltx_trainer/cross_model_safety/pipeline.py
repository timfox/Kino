"""Framework card, demos, smoke for cross-model safety steering."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cross_model_safety.config import CrossModelSafetyConfig, SteeringParams
from ltx_trainer.cross_model_safety.paper_tables import (
    table1_flux1_schnell_llama_svd,
    table1_native,
    table1_original,
    table1_transferred_svd,
    table2_mma_flux1_dev_svd,
)
from ltx_trainer.cross_model_safety.simulation import (
    alignment_comparison,
    alpha_sweep_demo,
    multi_vector_demo,
    random_baseline_asr,
    synthetic_setup,
)


def framework_card(cfg: CrossModelSafetyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CrossModelSafetyConfig()
    return {
        "name": "Cross-Model Safety Steering",
        "paper": cfg.paper_arxiv,
        "title": "Do Models Share Safety Representations? Cross-Model Steering for Safe Visual Generation",
        "pipeline": [
            "Estimate v_s from paired safe-unsafe prompts in source LLM",
            "Fit T_{s→t} on benign anchors only (SVD / ridge / MLP)",
            "Calibrate transferred v_t with anchor norm ratio β",
            "Steer target hidden states: ĥ = h + α v_t at inference",
        ],
        "source_llms": list(cfg.source_llms),
        "t2i_targets": list(cfg.t2i_targets),
        "t2v_targets": list(cfg.t2v_targets),
        "alignment_methods": list(cfg.alignment_methods),
        "safety_categories": list(cfg.safety_categories),
        "metrics": ["ASR (NudeNet + Q16)", "CLIP-Sim", "FID"],
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no Flux/Qwen/Wan pipelines or real NudeNet/Q16 detectors.",
        "Synthetic shared-linear geometry validates transfer ordering, not full I2P runs.",
        "Table 1–2 values are paper anchors; MLP is a lightweight ridge-like proxy.",
        "Multi-vector oracle weights use synthetic category labels only.",
        "T2V Wan2.2 results referenced via paper tables, not re-rendered here.",
    ]


def evaluation_demo(cfg: CrossModelSafetyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CrossModelSafetyConfig()
    params = cfg.params
    setup = synthetic_setup(n_anchors=params.anchor_count, seed=7)
    sweep = alpha_sweep_demo(setup, "svd")
    align = alignment_comparison(setup, alpha=params.default_alpha, params=params)
    multi = multi_vector_demo(setup, "ridge", alpha=3.0)
    rand_asr = random_baseline_asr(setup, alpha=params.default_alpha)
    original_asr = float(table1_original()["Flux1-Schnell"]["asr"])
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "tables": {
            "table1_original_flux_schnell": table1_original()["Flux1-Schnell"],
            "table1_native": table1_native(),
            "table1_transferred_llama_svd": table1_flux1_schnell_llama_svd(),
            "table1_highlights": {
                f"{k[0]}|{k[1]}": v for k, v in table1_transferred_svd().items()
            },
            "table2_mma_flux_dev_svd": table2_mma_flux1_dev_svd(),
        },
        "alpha_sweep_svd": sweep,
        "alignment_at_alpha5": align,
        "multi_vector_ridge": multi,
        "synthetic_random_asr": rand_asr,
        "synthetic_original_asr_anchor": original_asr,
    }


def evaluation_smoke(cfg: CrossModelSafetyConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    t1 = demo["tables"]["table1_transferred_llama_svd"]
    orig = demo["tables"]["table1_original_flux_schnell"]
    align = demo["alignment_at_alpha5"]
    sweep = demo["alpha_sweep_svd"]
    multi = demo["multi_vector_ridge"]

    assert abs(t1["asr"] - 0.038) < 0.001
    assert abs(t1["clip_sim"] - 0.308) < 0.001
    assert align["svd"]["asr"] < orig["asr"]
    assert align["svd"]["asr"] < demo["synthetic_random_asr"]
    assert align["ridge"]["clip_sim"] >= align["svd"]["clip_sim"] - 0.05
    assert sweep[-1]["asr"] < sweep[1]["asr"]  # α=7 vs α=0
    assert multi["multi_asr"] <= multi["global_asr"] + 0.05

    return {
        "status": "ok",
        "paper": (cfg or CrossModelSafetyConfig()).paper_arxiv,
        "table1_asr": t1["asr"],
        "svd_asr_synthetic": align["svd"]["asr"],
        "demo_keys": list(demo.keys()),
    }
