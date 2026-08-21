"""Synthetic timing curves aligned with Table 3 (stub)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.oma_fme.analysis_reuse import simulate_default_erp_times, simulate_ladder_times
from ltx_trainer.oma_fme.benchmarks import TABLE3_CMP_PRA_HQ_AVG, TABLE3_ERP_CRC_AVG
from ltx_trainer.oma_fme.metrics import delta_t_parallel, delta_t_serial, speedup_factor
from ltx_trainer.oma_fme.variants import ProjectionFormat, SharingStrategy


def _scale_to_target(
    measured: dict[str, float],
    target_delta_serial: float,
    ref: dict[str, float],
) -> dict[str, float]:
    """Scale heuristic times so ΔT_S matches paper anchor row."""
    ref_serial = ref["serial_s"]
    want = ref_serial * (1.0 + target_delta_serial / 100.0)
    scale = want / max(measured["serial_s"], 1e-6)
    return {
        "serial_s": measured["serial_s"] * scale,
        "parallel_max_s": measured["parallel_max_s"] * scale,
    }


def synthetic_variant_comparison() -> dict[str, Any]:
    ref = simulate_default_erp_times()
    erp_crc = simulate_ladder_times(SharingStrategy.CRC, ProjectionFormat.ERP)
    erp_crc = _scale_to_target(erp_crc, TABLE3_ERP_CRC_AVG["delta_T_S_pct"], ref)

    cmp_pra = simulate_ladder_times(SharingStrategy.PRA, ProjectionFormat.CMP)
    cmp_pra = _scale_to_target(cmp_pra, TABLE3_CMP_PRA_HQ_AVG["delta_T_S_pct"], ref)

    return {
        "ERP-Default": ref,
        "ERP-CRC": erp_crc,
        "CMP-PRA-HQ": cmp_pra,
        "delta_T_S": {
            "ERP-CRC": delta_t_serial([erp_crc["serial_s"]], [ref["serial_s"]]),
            "CMP-PRA-HQ": delta_t_serial([cmp_pra["serial_s"]], [ref["serial_s"]]),
        },
        "delta_T_P": {
            "ERP-CRC": delta_t_parallel([erp_crc["parallel_max_s"]], [ref["parallel_max_s"]]),
            "CMP-PRA-HQ": delta_t_parallel([cmp_pra["parallel_max_s"]], [ref["parallel_max_s"]]),
        },
        "speedup_x": {
            "ERP-CRC": speedup_factor(erp_crc["serial_s"], ref["serial_s"]),
            "CMP-PRA-HQ": speedup_factor(cmp_pra["serial_s"], ref["serial_s"]),
        },
    }
