"""Literature anchors from Section IV–V (not re-trained here)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.timesfm_attack.config import IEEE14BusAnchors, MassSpringConfig


def operating_guidelines() -> list[str]:
    return [
        "Deploy TimesFM gs as Layer-2 beside model-based gp (Swiss-cheese defense).",
        "Warmup Tw then clean window Tc before estimating residual covariance Σ̂.",
        "Update context buffer only when gs ≤ τs (Algorithm 1 line 12).",
        "Per-channel univariate TimesFM — no explicit cross-sensor coupling in forecast.",
        "Replace corrupted measurements with TimesFM point predictions after alarm for mitigation.",
        "Avoid high dropout — corrupts AP-style covariance buffers (analogous caution).",
    ]


def table_mass_spring_linear() -> dict[str, Any]:
    """Fig. 2 anchors (Section IV-A)."""
    cfg = MassSpringConfig()
    return {
        "plant": "undamped mass-spring, m=3 sensors",
        "alpha_p": cfg.alpha_p,
        "tau_p": 12.838,
        "L": cfg.context_length,
        "Tw": cfg.warmup_length,
        "Tc": cfg.clean_length,
        "theta": cfg.buffer_theta,
        "nominal_far_timesfm_pct": 0.5,
        "nominal_far_observer_pct": 2.8,
        "replay_ka": cfg.replay_ka,
        "replay_first_timesfm_alarm_k": 91,
        "stealthy_ka": cfg.stealthy_ka,
        "stealthy_delta_tau_p": cfg.delta_tau_p,
        "adaptive_attack_state_dev_reduction_x": 91.0,
    }


def table_ieee14_bus() -> dict[str, Any]:
    a = IEEE14BusAnchors()
    return {
        "buses": a.buses,
        "outputs": a.m_outputs,
        "alpha_p_pct": a.alpha_p * 100.0,
        "L": a.context_length,
        "Tw": a.warmup_length,
        "Tc": a.clean_length,
        "replay_ka": a.replay_ka,
        "stealthy_delta_tau_p": a.delta_tau_p,
        "stealthy_ka": a.stealthy_ka,
        "stealthy_k2": a.stealthy_k2,
        "ekf_stealthy_alarms_under_attack": 0,
        "timesfm_detects_stealthy": True,
    }


def table_mitigation_single_sensor() -> dict[str, Any]:
    return {
        "setting": "C=[1 0], single sensor under replay",
        "timesfm_protected_lower_error": True,
        "fundamental_limit_N_minus_2M": "fails — TimesFM substitution heuristic",
    }
