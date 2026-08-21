"""CCLab framework card, benchmark tables, and demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.cclab.config import (
    CCLabConfig,
    ENV_LEVEL_CC,
    FEATURE_LEVEL_CC,
    LEARNING_CC,
    NON_LEARNING_CC,
)
from ltx_trainer.cclab.metrics import bandwidth_utilization, cwnd_smoothness
from ltx_trainer.cclab.perturbations import append_bandwidth_step, perturb_min_rtt
from ltx_trainer.cclab.rewards import cc_reward, improved_adversarial_reward, queuing_delay


def framework_card(cfg: CCLabConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CCLabConfig()
    return {
        "name": "CCLab",
        "paper": "arXiv:2605.21915",
        "title": "Adversarial Testing of Learning- and Non-Learning-Based Congestion Controllers",
        "emulator": "Mahimahi",
        "manipulation_modes": ["feature_level_min_rtt", "environment_level_bandwidth"],
        "feature_level_cc": list(FEATURE_LEVEL_CC),
        "environment_level_cc": list(ENV_LEVEL_CC),
        "learning_cc": list(LEARNING_CC),
        "non_learning_cc": list(NON_LEARNING_CC),
        "link_delay_ms": cfg.link_delay_ms,
        "queue_bdp_multiplier": cfg.queue_bdp_multiplier,
        "metrics": ["bandwidth_utilization_pct", "queuing_delay_ms", "p95_delay_ms"],
    }


def table_feature_5pct() -> dict[str, dict[str, dict[str, float]]]:
    """Table I — minRTT perturbation [0.95, 1.05]."""
    return {
        "orca": {
            "clean": {"util_pct": 81.36, "delay_ms": 4.31, "p95_delay_ms": 12.51},
            "adversarial": {"util_pct": 80.48, "delay_ms": 4.50, "p95_delay_ms": 19.00},
        },
        "canopy": {
            "clean": {"util_pct": 94.32, "delay_ms": 19.21, "p95_delay_ms": 45.53},
            "adversarial": {"util_pct": 94.22, "delay_ms": 20.30, "p95_delay_ms": 46.61},
        },
        "bbr": {
            "clean": {"util_pct": 81.46, "delay_ms": 6.04, "p95_delay_ms": 22.17},
            "adversarial": {"util_pct": 76.94, "delay_ms": 6.65, "p95_delay_ms": 26.04},
        },
        "vegas": {
            "clean": {"util_pct": 70.94, "delay_ms": 2.91, "p95_delay_ms": 6.45},
            "adversarial": {"util_pct": 70.82, "delay_ms": 2.97, "p95_delay_ms": 6.50},
        },
    }


def table_feature_50pct() -> dict[str, dict[str, dict[str, float]]]:
    """Table II — minRTT perturbation [0.5, 1.5]."""
    return {
        "orca": {
            "clean": {"util_pct": 81.36, "delay_ms": 4.31, "p95_delay_ms": 12.51},
            "adversarial": {"util_pct": 73.67, "delay_ms": 4.23, "p95_delay_ms": 16.80},
        },
        "canopy": {
            "clean": {"util_pct": 94.32, "delay_ms": 19.21, "p95_delay_ms": 45.53},
            "adversarial": {"util_pct": 91.70, "delay_ms": 22.22, "p95_delay_ms": 49.47},
        },
        "bbr": {
            "clean": {"util_pct": 81.46, "delay_ms": 6.04, "p95_delay_ms": 22.17},
            "adversarial": {"util_pct": 66.78, "delay_ms": 6.25, "p95_delay_ms": 26.78},
        },
        "vegas": {
            "clean": {"util_pct": 70.94, "delay_ms": 2.91, "p95_delay_ms": 6.45},
            "adversarial": {"util_pct": 60.33, "delay_ms": 2.74, "p95_delay_ms": 4.42},
        },
    }


def feature_degradation_summary(table: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    """Utilization drop (%) per policy."""
    out: dict[str, float] = {}
    for policy, rows in table.items():
        clean = rows["clean"]["util_pct"]
        adv = rows["adversarial"]["util_pct"]
        out[policy] = max(0.0, clean - adv)
    return out


def table_environment_bandwidth() -> dict[str, dict[str, str]]:
    """Table III — utilization% / delay_ms per (target trace, tested policy)."""
    policies = ("orca", "cubic", "vegas", "canopy", "illinois", "tcp_lp")
    # Rows: adversarial agent trained against target; values are "util/delay"
    data: dict[str, dict[str, str]] = {
        "orca": {
            "orca": "85.49/61.54",
            "cubic": "95.37/51.25",
            "vegas": "66.68/26.25",
            "canopy": "87.67/68.88",
            "illinois": "95.84/55.17",
            "tcp_lp": "90.52/40.97",
        },
        "cubic": {
            "orca": "85.53/72.42",
            "cubic": "86.79/43.91",
            "vegas": "50.08/17.23",
            "canopy": "88.46/79.48",
            "illinois": "90.14/46.07",
            "tcp_lp": "80.16/33.24",
        },
        "vegas": {
            "orca": "90.88/18.49",
            "cubic": "75.68/15.41",
            "vegas": "48.42/7.48",
            "canopy": "84.78/20.08",
            "illinois": "80.25/15.67",
            "tcp_lp": "61.81/13.14",
        },
        "canopy": {
            "orca": "90.11/15.61",
            "cubic": "76.70/13.14",
            "vegas": "49.96/7.36",
            "canopy": "82.72/17.76",
            "illinois": "83.27/14.87",
            "tcp_lp": "55.95/10.36",
        },
        "illinois": {
            "orca": "83.45/22.41",
            "cubic": "81.04/19.80",
            "vegas": "55.41/10.75",
            "canopy": "85.37/25.19",
            "illinois": "83.60/20.78",
            "tcp_lp": "72.82/15.48",
        },
        "tcp_lp": {
            "orca": "93.36/17.38",
            "cubic": "87.06/14.16",
            "vegas": "54.27/5.61",
            "canopy": "85.31/20.46",
            "illinois": "92.41/17.39",
            "tcp_lp": "75.40/10.26",
        },
    }
    random_baseline = {
        "orca": "95.58/12.65",
        "cubic": "94.50/12.26",
        "vegas": "65.16/2.23",
        "canopy": "89.69/17.06",
        "illinois": "98.69/15.40",
        "tcp_lp": "85.58/6.77",
    }
    return {"targets": data, "random_baseline": random_baseline, "policies": list(policies)}


def environment_degradation_vs_random(cfg: CCLabConfig | None = None) -> dict[str, float]:
    """Worst-case utilization drop (pp) vs random baseline per policy column (Table III)."""
    cfg = cfg or CCLabConfig()
    tab = table_environment_bandwidth()
    rb = tab["random_baseline"]
    out: dict[str, float] = {}
    for policy in tab["policies"]:
        base = float(rb[policy].split("/")[0])
        worst_util = base
        for _target, row in tab["targets"].items():
            util = float(row[policy].split("/")[0])
            worst_util = min(worst_util, util)
        out[policy] = max(0.0, base - worst_util)
    _ = cfg  # Paper headline Sec. V-D: Orca ~12%, Canopy ~7%, Cubic 19%, LP ~30%
    return out


def table_cwnd_smoothness() -> dict[str, dict[str, float]]:
    """Table IV — cwnd smoothness on Canopy-targeted adversarial trace."""
    return {
        "orca": {"cwnd_smoothness": 6544.59, "cwnd_smoothness_log": 7.13},
        "canopy": {"cwnd_smoothness": 7149.33, "cwnd_smoothness_log": 7.34},
        "cubic": {"cwnd_smoothness": 180.13, "cwnd_smoothness_log": 2.15},
        "vegas": {"cwnd_smoothness": 157.85, "cwnd_smoothness_log": 2.28},
        "illinois": {"cwnd_smoothness": 951.42, "cwnd_smoothness_log": 4.46},
        "tcp_lp": {"cwnd_smoothness": 900.11, "cwnd_smoothness_log": 4.33},
    }


def table_adversarial_training() -> dict[str, dict[str, float]]:
    """Table V — Orca before/after adversarial training."""
    return {
        "random_baseline_no_retrain": {"util_pct": 95.58, "delay_ms": 12.65},
        "random_baseline_adv_train": {"util_pct": 94.65, "delay_ms": 12.86},
        "synthetic_no_retrain": {"util_pct": 94.11, "delay_ms": 14.18},
        "synthetic_adv_train": {"util_pct": 95.12, "delay_ms": 21.03},
        "real_world_no_retrain": {"util_pct": 83.21, "delay_ms": 29.41},
        "real_world_adv_train": {"util_pct": 83.89, "delay_ms": 30.09},
        "adversarial_no_retrain": {"util_pct": 85.49, "delay_ms": 61.54},
        "adversarial_adv_train": {"util_pct": 91.66, "delay_ms": 57.83},
        "canopy_adversarial_no_retrain": {"util_pct": 86.67, "delay_ms": 68.99},
    }


def adversarial_step_demo(cfg: CCLabConfig | None = None) -> dict[str, float]:
    """Synthetic closed-loop step: CC reward + improved adversarial reward."""
    cfg = cfg or CCLabConfig()
    rtt_min = torch.tensor(10.0)
    rtt = torch.tensor(14.0)
    perturbed = perturb_min_rtt(rtt_min, 1.25, low=0.5, high=1.5)
    delays = queuing_delay(torch.tensor([12.0, 13.0, 14.0, 15.0]), rtt_min)
    r_cc = cc_reward(
        throughput=torch.tensor(80.0),
        loss_rate=torch.tensor(0.01),
        rtt=rtt,
        rtt_min=perturbed,
        bmax=torch.tensor(100.0),
        cfg=cfg,
    )
    util = bandwidth_utilization(72.0, 96.0)
    r_adv = improved_adversarial_reward(
        torch.tensor(util),
        delays,
        tau=float(delays.mean()),
        cfg=cfg,
    )
    trace = torch.tensor([40.0, 50.0])
    trace2, bw = append_bandwidth_step(
        trace,
        90.0,
        delta_budget=cfg.smoothness_budget_delta_mbps,
        window_k=cfg.smoothness_window_k,
        bw_min=cfg.bw_min_mbps,
        bw_max=cfg.bw_max_mbps,
    )
    cwnd = torch.tensor([100.0, 150.0, 220.0, 180.0])
    return {
        "cc_reward": float(r_cc.item()),
        "adversarial_reward": float(r_adv.item()),
        "perturbed_min_rtt_ms": float(perturbed.item()),
        "utilization_pct": util,
        "trace_len": float(trace2.numel()),
        "last_bw_mbps": bw,
        **cwnd_smoothness(cwnd),
    }
