"""Experimental kernel identification — diffusivity sweep (Fig. 3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.cal_diffusion.config import DiffusionSweep, VoxelConfig
from ltx_trainer.cal_diffusion.metrics import assd_um, msssim_stub, thickness_map
from ltx_trainer.cal_diffusion.tomography import equal_volume_binarize, forward_dose, plate_target_volume


@dataclass(frozen=True)
class IdentificationResult:
    d_m2_s: float
    assd_um: float
    ms_ssim: float


def simulate_uncorrected_print(
    target: np.ndarray,
    *,
    true_d_m2_s: float,
    t_s: float,
    voxel_m: float,
) -> np.ndarray:
    dose = forward_dose(target, d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    vox = int((target > 0.5).sum())
    return equal_volume_binarize(dose, vox)


def identify_diffusivity(
    sweep: DiffusionSweep | None = None,
    vox_cfg: VoxelConfig | None = None,
    *,
    true_d_m2_s: float | None = None,
) -> list[IdentificationResult]:
    """Sweep D and score model vs synthetic micro-CT (uncorrected print)."""
    sweep = sweep or DiffusionSweep()
    vox_cfg = vox_cfg or VoxelConfig()
    true_d = true_d_m2_s if true_d_m2_s is not None else sweep.notional_d_m2_s
    target = plate_target_volume(vox_cfg.volume_shape)
    ct = simulate_uncorrected_print(
        target,
        true_d_m2_s=true_d,
        t_s=sweep.uncorrected_print_time_s,
        voxel_m=vox_cfg.voxel_m,
    )
    tm_ct = thickness_map(ct)
    results: list[IdentificationResult] = []
    for d in sweep.sweep_d_m2_s:
        dose = forward_dose(
            target,
            d_m2_s=d,
            t_s=sweep.uncorrected_print_time_s,
            voxel_m=vox_cfg.voxel_m,
        )
        vox = int((target > 0.5).sum())
        model = equal_volume_binarize(dose, vox)
        results.append(
            IdentificationResult(
                d_m2_s=d,
                assd_um=assd_um(model, ct, voxel_um=vox_cfg.voxel_um),
                ms_ssim=msssim_stub(thickness_map(model), tm_ct),
            )
        )
    return results


def best_identified_d(results: list[IdentificationResult]) -> float:
    """D minimizing ASSD (Fig. 3)."""
    return min(results, key=lambda r: r.assd_um).d_m2_s
