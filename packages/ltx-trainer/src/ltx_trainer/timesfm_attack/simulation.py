"""Monte-Carlo-lite experiments (Section IV-A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.timesfm_attack.attacks import apply_replay_attack, stealthy_attack_lti
from ltx_trainer.timesfm_attack.config import MassSpringConfig
from ltx_trainer.timesfm_attack.detector import run_timesfm_detector
from ltx_trainer.timesfm_attack.observer import luenberger_gain, run_primary_detector
from ltx_trainer.timesfm_attack.plant import mass_spring_matrices, simulate_mass_spring


@dataclass
class AttackScenarioResult:
    name: str
    primary_far_nominal: float
    timesfm_far_nominal: float
    primary_alarm_rate_attack: float
    timesfm_alarm_rate_attack: float
    primary_sustained_steps: int
    timesfm_sustained_steps: int
    first_timesfm_alarm: int | None


def _far(alarms: np.ndarray, start: int, end: int) -> float:
    seg = alarms[start:end]
    if len(seg) == 0:
        return 0.0
    return float(np.mean(seg))


def _attack_rate(alarms: np.ndarray, ka: int, end: int | None = None) -> float:
    end = len(alarms) if end is None else end
    return _far(alarms, ka, end)


def _sustained(alarms: np.ndarray, ka: int) -> int:
    count = 0
    for t in range(ka, len(alarms)):
        if alarms[t]:
            count += 1
        else:
            count = 0
        if count >= 3:
            return t - 2
    return -1


def run_mass_spring_scenario(
    cfg: MassSpringConfig,
    *,
    attack: str,
) -> AttackScenarioResult:
    _, y = simulate_mass_spring(cfg)
    a, c = mass_spring_matrices(cfg)
    k = luenberger_gain(a, c, cfg.observer_poles)

    primary_nom = run_primary_detector(y, cfg)
    timesfm_nom = run_timesfm_detector(y, cfg)

    if attack == "replay":
        y_att = apply_replay_attack(y, cfg)
        ka = cfg.replay_ka
        k_end = len(y)
    elif attack == "stealthy":
        y_att = stealthy_attack_lti(
            y,
            cfg,
            sigma_p=primary_nom.sigma_p,
            k_gain=k,
            tau_p=primary_nom.tau,
        )
        ka = cfg.stealthy_ka
        k_end = cfg.stealthy_k2 + 1
    else:
        raise ValueError(f"unknown attack: {attack}")

    primary_att = run_primary_detector(y, cfg, y_tilde=y_att)
    timesfm_att = run_timesfm_detector(y_att, cfg)

    nom_end = ka
    first_tf = next((t for t in range(ka, len(timesfm_att.alarms)) if timesfm_att.alarms[t]), None)

    return AttackScenarioResult(
        name=attack,
        primary_far_nominal=_far(primary_nom.alarms, cfg.warmup_length, nom_end),
        timesfm_far_nominal=_far(timesfm_nom.alarms, cfg.warmup_length + cfg.clean_length, nom_end),
        primary_alarm_rate_attack=_attack_rate(primary_att.alarms, ka, k_end),
        timesfm_alarm_rate_attack=_attack_rate(timesfm_att.alarms, ka),
        primary_sustained_steps=max(_sustained(primary_att.alarms, ka), -1),
        timesfm_sustained_steps=max(_sustained(timesfm_att.alarms, ka), -1),
        first_timesfm_alarm=first_tf,
    )


def run_mass_spring_suite(cfg: MassSpringConfig | None = None) -> dict[str, AttackScenarioResult]:
    cfg = cfg or MassSpringConfig()
    return {
        "replay": run_mass_spring_scenario(cfg, attack="replay"),
        "stealthy": run_mass_spring_scenario(cfg, attack="stealthy"),
    }
