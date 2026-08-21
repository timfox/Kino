"""ICBHI and SPRSound device registry (Table 1, arXiv:2605.29862)."""

from __future__ import annotations

from typing import Any


def device_registry() -> list[dict[str, Any]]:
    """Table 1 — device-wise statistics and label ratios."""
    return [
        {
            "device": "AKGC417L",
            "dataset": "ICBHI",
            "mean_mu_d": -51.31,
            "mean_sf": 3275.76,
            "mean_st": 47239.05,
            "subjects": 31,
            "labels": {"Normal": 0.454, "Crackle": 0.363, "Wheeze": 0.107, "Both": 0.076},
        },
        {
            "device": "LittC2SE",
            "dataset": "ICBHI",
            "mean_mu_d": -90.53,
            "mean_sf": 1540.72,
            "mean_st": 11640.13,
            "subjects": 21,
            "labels": {"Normal": 0.587, "Crackle": 0.115, "Wheeze": 0.220, "Both": 0.077},
        },
        {
            "device": "Litt3200",
            "dataset": "ICBHI",
            "mean_mu_d": -77.62,
            "mean_sf": 409.18,
            "mean_st": 1141.44,
            "subjects": 9,
            "labels": {"Normal": 0.713, "Crackle": 0.053, "Wheeze": 0.205, "Both": 0.029},
        },
        {
            "device": "Meditron",
            "dataset": "ICBHI",
            "mean_mu_d": -77.34,
            "mean_sf": 2487.34,
            "mean_st": 20810.60,
            "subjects": 61,
            "labels": {"Normal": 0.753, "Crackle": 0.132, "Wheeze": 0.101, "Both": 0.014},
        },
        {
            "device": "Yunting",
            "dataset": "SPRSound",
            "mean_mu_d": -82.73,
            "mean_sf": 388.91,
            "mean_st": 2102.70,
            "subjects": 284,
            "labels": {"Normal": 0.772, "Crackle": 0.130, "Wheeze": 0.094, "Both": 0.004},
        },
    ]


def lodo_splits() -> list[dict[str, Any]]:
    """Leave-one-device-out federated validation settings."""
    return [
        {
            "setting": 1,
            "name": "LODO AKGC417L/Meditron/Yunting",
            "train_devices": ["Meditron", "Yunting"],
            "held_out": "AKGC417L",
        },
        {
            "setting": 1,
            "name": "LODO Meditron",
            "train_devices": ["AKGC417L", "Yunting"],
            "held_out": "Meditron",
        },
        {
            "setting": 1,
            "name": "LODO Yunting",
            "train_devices": ["AKGC417L", "Meditron"],
            "held_out": "Yunting",
        },
        {
            "setting": 2,
            "name": "Littmann-family leave-out",
            "train_devices": ["AKGC417L", "Meditron", "Yunting"],
            "held_out": ["LittC2SE", "Litt3200"],
        },
    ]
