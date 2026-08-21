"""Framework card, demos, smoke for AHAD (arXiv:2606.05369)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ahad.config import AHADConfig
from ltx_trainer.ahad.criterion import shift_weights
from ltx_trainer.ahad.paper_tables import (
    dataset_catalog,
    remark2_thresholds,
    table_ii_average,
    table_iii_ablation_average,
)
import numpy as np

from ltx_trainer.ahad.criterion import estimate_robust_perturbation
from ltx_trainer.ahad.simulation import (
    ablation_table,
    crop_surrogate,
    evaluate_ahad_on_hsi,
    shift_robustness_demo,
    synthetic_hsi,
)


def framework_card(cfg: AHADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AHADConfig()
    p = cfg.params
    return {
        "name": "AHAD",
        "paper": cfg.paper_arxiv,
        "title": "Anti-hyperspectral anomaly detection via Lipschitz-forcing perturbations",
        "problem": "Estimate P so Y_A = Y_H + P evades unknown HAD detectors",
        "characteristics": ["black_box", "arab", "energy_efficient", "restoration_resistant"],
        "regularizers": ["SSTV (REG1)", "Lipschitz ARAB (Φ_W)", "pseudo-anomaly tail (PAG)"],
        "robustness": "Gaussian-weighted pixel shifts (Lemma 1 / Eq. 15–20)",
        "metric": "ArmCBA = (1 - AUC_ap/AUC_up) × 100%",
        "datasets": list(cfg.datasets),
        "had_baselines": list(cfg.had_baselines),
        "default_params": {
            "lambda1": p.lambda1,
            "lambda2": p.lambda2,
            "shift_radius": p.shift_radius,
            "sigma_shift": p.sigma_shift,
        },
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no PyTorch Adam on full HSIs or ten benchmark HAD re-implementations.",
        "Proxy RX/subspace detector substitutes for SuperRPCA, BockNet, OTAD, etc.",
        "ArmCBA table anchors from paper; synthetic ordering validates regularizer roles.",
        "Imperfect CSI modeled as discrete pixel shifts with Gaussian q(Δ).",
        "GLF denoising uses low-rank smoothing proxy, not full Zhuang et al. GLF.",
    ]


def evaluation_demo(cfg: AHADConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AHADConfig()
    params = cfg.params
    y_full, mask_full = synthetic_hsi(seed=7)
    r = params.shift_radius
    y = crop_surrogate(y_full, r)
    mask = crop_surrogate(mask_full.astype(float), r) > 0.5
    rng = np.random.default_rng(7)
    full = evaluate_ahad_on_hsi(y, mask, params, rng=rng)
    p = np.clip(estimate_robust_perturbation(y, params, rng=rng), -0.35, 0.35)
    robust = shift_robustness_demo(y, mask, p, params)
    ablation = ablation_table(cfg, seed=7)
    weights = shift_weights(params.shift_radius, params.sigma_shift)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "remark2": remark2_thresholds(),
        "datasets": dataset_catalog(),
        "tables": {
            "table_ii_average": table_ii_average(),
            "table_iii_ablation_average": table_iii_ablation_average(),
        },
        "synthetic_eval": full,
        "shift_robustness": robust,
        "shift_weights_sum": float(sum(weights.values())),
        "ablation": ablation,
    }


def evaluation_smoke(cfg: AHADConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    t2 = demo["tables"]["table_ii_average"]
    t3 = demo["tables"]["table_iii_ablation_average"]
    syn = demo["synthetic_eval"]
    ablation = {row["case"]: row["armcba"] for row in demo["ablation"]}

    assert abs(t2["armcba_pct"] - 15.8465) < 0.01
    assert abs(t3["case4_full"] - 16.3211) < 0.01
    assert abs(demo["shift_weights_sum"] - 1.0) < 1e-6
    assert syn["auc_ap"] < syn["auc_up"]
    assert syn["armcba"] > 5.0
    assert ablation["SSTV-Lipschitz-PAG"] > ablation["SSTV-PAG"]
    assert ablation["SSTV-Lipschitz"] > ablation["SSTV-PAG"]
    assert demo["shift_robustness"]["mean_armcba"] > 4.0

    return {
        "status": "ok",
        "paper": (cfg or AHADConfig()).paper_arxiv,
        "table_ii_armcba": t2["armcba_pct"],
        "synthetic_armcba": syn["armcba"],
        "ablation_full": ablation["SSTV-Lipschitz-PAG"],
        "demo_keys": list(demo.keys()),
    }
