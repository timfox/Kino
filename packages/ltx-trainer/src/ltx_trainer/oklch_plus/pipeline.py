"""Framework card, demos, smoke for Oklch+ (arXiv:2606.05255)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.oklch_plus.config import OklchPlusConfig, OklchPlusParams
from ltx_trainer.oklch_plus.metrics import (
    operating_guidelines,
    table1_chroma_functions,
    table2_overall_stress,
    table3_subdataset_stress,
    table4_attribute_stress,
    table5_cross_validation,
    table6_cam16_appearance,
    table7_gamut_cam16,
    table8_appearance_vs_discrimination,
)
from ltx_trainer.oklch_plus.simulation import (
    chroma_function_stress_on_synthetic,
    combvd_low_chroma_fraction,
    cross_validation_summary,
    interpolation_uniformity_demo,
    model_comparison,
    subdataset_improvement,
    synthetic_combvd_pairs,
    synthetic_stress_comparison,
    table1_vs_nr_advantage,
)
from ltx_trainer.oklch_plus.transforms import naka_rushton_chroma, sigmoid_chroma


def framework_card(cfg: OklchPlusConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OklchPlusConfig()
    p = cfg.params
    return {
        "name": "Oklch+",
        "paper": cfg.paper_arxiv,
        "title": "Three-parameter extension of Oklab for color difference prediction",
        "dataset": cfg.dataset,
        "stress_formula": cfg.anchor_formula,
        "parameters": {"alpha": p.alpha, "n": p.n, "sigma": p.sigma},
        "transforms": [
            "L′ = L^α",
            "C′ = C^n / (C^n + σ^n)",
            "ΔE = Euclidean distance in (L′, a′, b′)",
        ],
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "CPU stub — COMBVD pairs are synthetic; table STRESS values use paper anchors.",
        "No full CIEDE2000 implementation; CIEDE2000 STRESS is anchor-only.",
        "CAM16-UCS gamut tables (Tables 6–7) are reference anchors, not recomputed.",
        "Evaluation centered on low chroma (99.2% COMBVD pairs C < 0.20).",
        "HELMLAB STRESS uses a different normalization — not directly comparable.",
    ]


def evaluation_demo(cfg: OklchPlusConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OklchPlusConfig()
    pairs = synthetic_combvd_pairs()
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "synthetic": {
            "n_pairs": len(pairs),
            "low_chroma_fraction": combvd_low_chroma_fraction(pairs),
            "stress": synthetic_stress_comparison(),
        },
        "model_comparison": model_comparison(),
        "chroma_ablation": chroma_function_stress_on_synthetic(),
        "subdataset_gain": subdataset_improvement(),
        "interpolation": interpolation_uniformity_demo(),
        "cross_validation": cross_validation_summary(),
        "tables": {
            "chroma_functions": table1_chroma_functions(),
            "overall": table2_overall_stress(),
            "subdataset": table3_subdataset_stress(),
            "attributes": table4_attribute_stress(),
            "cross_val": table5_cross_validation(),
            "cam16_srgb": table6_cam16_appearance(),
            "gamuts": table7_gamut_cam16(),
            "appearance_reversal": table8_appearance_vs_discrimination(),
        },
    }


def evaluation_smoke(cfg: OklchPlusConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    overall = demo["tables"]["overall"]
    synth = demo["synthetic"]["stress"]
    cv = demo["cross_validation"]
    gains = demo["subdataset_gain"]

    assert overall["oklch_plus"]["stress"] == 29.09
    assert overall["ciede2000"]["stress"] == 29.13
    assert abs(overall["oklch_plus"]["stress"] - overall["ciede2000"]["stress"]) <= 0.05
    assert overall["oklch_plus"]["stress"] < overall["oklab"]["stress"]
    assert overall["oklch_plus"]["stress"] < overall["power_lc"]["stress"]
    assert synth["oklch_plus"] < synth["oklab"]
    assert cv["test_oklch_plus"] < cv["test_oklab"]
    assert abs(cv["test_oklch_plus"] - cv["test_ciede2000"]) <= 2.5
    assert all(v > 0 for v in gains.values())
    assert naka_rushton_chroma(np.array([0.0]), n=0.87, sigma=0.34)[0] == 0.0
    assert sigmoid_chroma(np.array([0.0]), a=42.0, cth=0.068)[0] > 0.0
    assert table1_vs_nr_advantage() > 0

    return {
        "status": "ok",
        "paper": (cfg or OklchPlusConfig()).paper_arxiv,
        "oklch_plus_stress": overall["oklch_plus"]["stress"],
        "ciede2000_gap": abs(overall["oklch_plus"]["stress"] - overall["ciede2000"]["stress"]),
        "demo_keys": list(demo.keys()),
    }
