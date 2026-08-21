"""GOPEX LTX / vLLM training energy budgeting hooks."""

from __future__ import annotations

import os
from typing import Any

from ltx_trainer.energy_aware.green_ai import estimate_training_cluster_mw
from ltx_trainer.energy_aware.metrics import carbon_kg_co2e, hourly_electricity_cost_usd
from ltx_trainer.energy_aware.profiling import profiling_report


def ltx_training_energy_plan(
    *,
    num_gpus: int | None = None,
    hours: float,
    tdp_w: float | None = None,
    utilization: float = 0.85,
    kg_co2_per_kwh: float | None = None,
    usd_per_kwh: float | None = None,
    dyna_pruner: bool = False,
) -> dict[str, Any]:
    """
    Estimate OpEx/carbon for an LTX training run.

    Env overrides: ``GOPEX_GPU_TDP_W``, ``GOPEX_GRID_CO2_KG_PER_KWH``,
    ``GOPEX_ELECTRICITY_USD_PER_KWH``, ``GOPEX_TRAIN_GPU_COUNT``,
    ``GOPEX_DYNA_PRUNER`` (set ``1`` to include co-pruning efficiency factors).
    """
    if not dyna_pruner:
        dyna_pruner = os.environ.get("GOPEX_DYNA_PRUNER", "").strip() in ("1", "true", "yes")
    gpus = num_gpus if num_gpus is not None else int(os.environ.get("GOPEX_TRAIN_GPU_COUNT", "1"))
    tdp = tdp_w if tdp_w is not None else float(os.environ.get("GOPEX_GPU_TDP_W", "700"))
    co2 = kg_co2_per_kwh if kg_co2_per_kwh is not None else float(os.environ.get("GOPEX_GRID_CO2_KG_PER_KWH", "0.35"))
    price = usd_per_kwh if usd_per_kwh is not None else float(os.environ.get("GOPEX_ELECTRICITY_USD_PER_KWH", "0.15"))

    mw = estimate_training_cluster_mw(gpus, tdp, utilization=utilization)
    power_w = mw * 1e6
    prof = profiling_report(power_w=power_w, duration_s=hours * 3600, kg_co2_per_kwh=co2)
    result: dict[str, Any] = {
        "gpus": gpus,
        "hours": hours,
        "sustained_mw": round(mw, 4),
        "electricity_usd": round(hourly_electricity_cost_usd(mw, price) * hours, 2),
        "carbon_kg_co2e": round(carbon_kg_co2e(mw, hours, co2), 2),
        "profiling": prof,
        "notes": "Use NVML/ALUMET for measured power; this is planning-grade.",
    }
    if dyna_pruner:
        from ltx_trainer.dyna_pruner.ltx_plan import energy_aware_training_note

        note = energy_aware_training_note()
        factor = float(note["effective_power_factor"])
        result["dyna_pruner"] = note
        result["adjusted_sustained_mw"] = round(mw * factor, 4)
        result["adjusted_electricity_usd"] = round(result["electricity_usd"] * factor, 2)
        result["adjusted_carbon_kg_co2e"] = round(result["carbon_kg_co2e"] * factor, 2)
    return result
