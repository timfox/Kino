"""Framework card, demos, smoke (arXiv:2606.06347)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.timesfm_attack.chi2 import attack_budget_delta_tau, chi2_threshold
from ltx_trainer.timesfm_attack.config import MassSpringConfig, TimesFMAttackConfig
from ltx_trainer.timesfm_attack.metrics import (
    operating_guidelines,
    table_ieee14_bus,
    table_mass_spring_linear,
    table_mitigation_single_sensor,
)
from ltx_trainer.timesfm_attack.observer import luenberger_gain, run_primary_detector
from ltx_trainer.timesfm_attack.plant import mass_spring_matrices, simulate_mass_spring
from ltx_trainer.timesfm_attack.simulation import run_mass_spring_suite
from ltx_trainer.timesfm_attack.surrogate import timesfm_predict


def framework_card(cfg: TimesFMAttackConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TimesFMAttackConfig()
    return {
        "name": "TimesFM-CPS-Attack",
        "paper": cfg.paper_arxiv,
        "title": "Attack detection using time-series foundation models",
        "layers": ["primary gp (model-based χ²)", "secondary gs (TimesFM χ²)"],
        "attacks": ["model-free replay", "model-based stealthy (Thm 1/2)"],
        "plant_knowledge": "none for gs — zero-shot TimesFM surrogate",
        "benchmarks": ["mass-spring LTI", "IEEE 14-bus swing"],
        "code": "https://github.com/balajianand1994/Attack_detection_using_TFM.git",
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "This stub uses AR(2) per-channel forecast, not the real TimesFM checkpoint.",
        "Online gs uses open-loop counterfactual forecasts (frozen coefs) so replay cannot",
        "contaminate the surrogate residual path — mimics TimesFM lacking closed-form innovations.",
        "Theorem 2 EKF stealthy attack is not fully reimplemented — LTI Thm 1 only in demo.",
        "IEEE 14-bus nonlinear swing dynamics are anchor-only, not simulated here.",
        "SPSA adaptive attacker (Sec IV-A.4) reported empirically, not reproduced.",
        "Univariate TimesFM ignores cross-channel coupling present in CPS measurements.",
    ]


def evaluation_demo(cfg: TimesFMAttackConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TimesFMAttackConfig()
    ms_cfg = MassSpringConfig()
    _, y = simulate_mass_spring(ms_cfg)
    a, c = mass_spring_matrices(ms_cfg)
    k = luenberger_gain(a, c, ms_cfg.observer_poles)
    m = c.shape[0]

    primary = run_primary_detector(y, ms_cfg)
    tau_s = chi2_threshold(m, ms_cfg.alpha_s)
    delta_tau = attack_budget_delta_tau(m, primary.tau, ms_cfg.alpha_p, 0.0)

    hist = y[ms_cfg.warmup_length - ms_cfg.context_length : ms_cfg.warmup_length]
    y_pred = timesfm_predict(hist)

    scenarios = run_mass_spring_suite(ms_cfg)

    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "chi2": {
            "m": m,
            "tau_p": primary.tau,
            "tau_s": tau_s,
            "delta_tau_budget": delta_tau,
            "configured_delta_tau_attack": ms_cfg.delta_tau_p,
        },
        "surrogate_preview": {
            "context_shape": list(hist.shape),
            "prediction_shape": list(y_pred.shape),
        },
        "scenarios": {k: v.__dict__ for k, v in scenarios.items()},
        "tables": {
            "mass_spring": table_mass_spring_linear(),
            "ieee14": table_ieee14_bus(),
            "mitigation": table_mitigation_single_sensor(),
        },
    }


def evaluation_smoke(cfg: TimesFMAttackConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    replay = demo["scenarios"]["replay"]
    stealthy = demo["scenarios"]["stealthy"]

    assert replay["timesfm_alarm_rate_attack"] > replay["primary_alarm_rate_attack"]
    assert replay["timesfm_sustained_steps"] >= 0
    assert stealthy["timesfm_alarm_rate_attack"] > 0.15
    assert stealthy["primary_alarm_rate_attack"] <= max(0.25, stealthy["primary_far_nominal"] * 3.0)

    return {
        "status": "ok",
        "paper": (cfg or TimesFMAttackConfig()).paper_arxiv,
        "demo_keys": list(demo.keys()),
    }
