"""Paper stub smoke for Energy-Aware Computing 2026 survey module."""

from __future__ import annotations

from typing import Any

from ltx_trainer.energy_aware.cooling import rack_requires_liquid_cooling
from ltx_trainer.energy_aware.green_ai import estimate_llm_inference_joules, estimate_training_cluster_mw
from ltx_trainer.energy_aware.hardware import exascale_table_dict
from ltx_trainer.energy_aware.ltx_training import ltx_training_energy_plan
from ltx_trainer.energy_aware.metrics import carbon_kg_co2e, power_usage_effectiveness
from ltx_trainer.energy_aware.scheduling import carbon_aware_start_hour, first_fit_bin_pack
from ltx_trainer.energy_aware.scheduling import CarbonWindow
from ltx_trainer.energy_aware.taxonomy import taxonomy_dict


def evaluation_smoke() -> dict[str, Any]:
    windows = [CarbonWindow(h, 0.2 + 0.3 * (h % 12) / 12) for h in range(24)]
    start = carbon_aware_start_hour(4.0, windows)
    bins, n_bins = first_fit_bin_pack([0.4, 0.3, 0.5, 0.2, 0.6], 1.0)
    infer = estimate_llm_inference_joules(
        prefill_w=400.0,
        prefill_s=0.05,
        decode_w=250.0,
        decode_s_per_token=0.012,
        num_tokens=128,
    )
    jupiter = next(r for r in exascale_table_dict() if r["machine"] == "JUPITER")
    return {
        "package": "energy_aware",
        "paper": "arXiv:2605.24569",
        "taxonomy_pillars": len(taxonomy_dict()["pillars"]),
        "carbon_aware_start_hour": start,
        "vm_bins": n_bins,
        "vm_bin_assignments": bins,
        "pue_sample": round(power_usage_effectiveness(1.0, 1.25), 3),
        "jupiter_kg_co2_per_hour": jupiter["kg_co2_per_hour"],
        "training_mw_8gpu": round(estimate_training_cluster_mw(8, 700.0), 4),
        "inference_joules_per_token": round(infer["joules_per_token"], 4),
        "liquid_cooling_120kw_rack": rack_requires_liquid_cooling(120.0),
        "ltx_plan_24h": ltx_training_energy_plan(num_gpus=1, hours=24.0),
        "frontier_carbon_1h": round(carbon_kg_co2e(22.78, 1.0, 0.36), 0),
    }
