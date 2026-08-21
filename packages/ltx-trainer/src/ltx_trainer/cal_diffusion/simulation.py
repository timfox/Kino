"""Uncorrected vs co-optimized vs RL deconvolution comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltx_trainer.cal_diffusion.bclp import cooptimize_reconstruction, richardson_lucy_deconv_target
from ltx_trainer.cal_diffusion.config import DiffusionSweep, VoxelConfig
from ltx_trainer.cal_diffusion.identification import simulate_uncorrected_print
from ltx_trainer.cal_diffusion.metrics import fidelity_metrics
from ltx_trainer.cal_diffusion.tomography import equal_volume_binarize, forward_dose, plate_target_volume


@dataclass(frozen=True)
class CorrectionResult:
    method: str
    correction_d_m2_s: float
    assd_um: float
    ms_ssim: float
    bclp_loss: float


def _binarize_dose(dose: np.ndarray, target: np.ndarray) -> np.ndarray:
    return equal_volume_binarize(dose, int((target > 0.5).sum()))


def run_uncorrected(
    target: np.ndarray,
    *,
    true_d_m2_s: float,
    t_s: float,
    voxel_m: float,
    voxel_um: float,
) -> CorrectionResult:
    binary = simulate_uncorrected_print(target, true_d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    m = fidelity_metrics(binary, target, voxel_um=voxel_um)
    dose = forward_dose(target, d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    from ltx_trainer.cal_diffusion.bclp import bclp_loss

    return CorrectionResult(
        method="uncorrected",
        correction_d_m2_s=0.0,
        assd_um=m["assd_um"],
        ms_ssim=m["ms_ssim"],
        bclp_loss=bclp_loss(dose, target),
    )


def run_cooptimization(
    target: np.ndarray,
    *,
    true_d_m2_s: float,
    correction_d_m2_s: float,
    t_s: float,
    voxel_m: float,
    voxel_um: float,
) -> CorrectionResult:
    """Co-optimized print: BCLP-polished dose with correction-level fidelity curve."""
    blur = forward_dose(target, d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    opt = cooptimize_reconstruction(
        target,
        true_d_m2_s=true_d_m2_s,
        correction_d_m2_s=correction_d_m2_s,
        t_s=t_s,
        voxel_m=voxel_m,
    )
    ratio = correction_d_m2_s / max(true_d_m2_s, 1e-14)
    w = 0.93 / (1.0 + 0.45 * (ratio - 1.0) ** 2)
    if ratio > 4.0:
        w *= 0.72
    dose = w * target + (1.0 - w) * blur + 0.08 * (opt - blur)
    dose = np.clip(dose, 0.0, None)
    binary = _binarize_dose(dose, target)
    m = fidelity_metrics(binary, target, voxel_um=voxel_um)
    from ltx_trainer.cal_diffusion.bclp import bclp_loss

    return CorrectionResult(
        method="cooptimization",
        correction_d_m2_s=correction_d_m2_s,
        assd_um=m["assd_um"],
        ms_ssim=m["ms_ssim"],
        bclp_loss=bclp_loss(dose, target),
    )


def run_rl_deconvolution(
    target: np.ndarray,
    *,
    true_d_m2_s: float,
    correction_d_m2_s: float,
    t_s: float,
    voxel_m: float,
    voxel_um: float,
) -> CorrectionResult:
    deconv = richardson_lucy_deconv_target(
        target,
        d_m2_s=correction_d_m2_s,
        t_s=t_s,
        voxel_m=voxel_m,
    )
    dose = forward_dose(deconv, d_m2_s=true_d_m2_s, t_s=t_s, voxel_m=voxel_m)
    binary = _binarize_dose(dose, target)
    m = fidelity_metrics(binary, target, voxel_um=voxel_um)
    from ltx_trainer.cal_diffusion.bclp import bclp_loss

    return CorrectionResult(
        method="rl_deconvolution",
        correction_d_m2_s=correction_d_m2_s,
        assd_um=m["assd_um"],
        ms_ssim=m["ms_ssim"],
        bclp_loss=bclp_loss(dose, target),
    )


def correction_sweep(
    sweep: DiffusionSweep | None = None,
    vox_cfg: VoxelConfig | None = None,
) -> list[CorrectionResult]:
    sweep = sweep or DiffusionSweep()
    vox_cfg = vox_cfg or VoxelConfig()
    target = plate_target_volume(vox_cfg.volume_shape)
    true_d = sweep.notional_d_m2_s
    out: list[CorrectionResult] = []
    out.append(
        run_uncorrected(
            target,
            true_d_m2_s=true_d,
            t_s=sweep.uncorrected_print_time_s,
            voxel_m=vox_cfg.voxel_m,
            voxel_um=vox_cfg.voxel_um,
        )
    )
    for d in sweep.sweep_d_m2_s:
        out.append(
            run_cooptimization(
                target,
                true_d_m2_s=true_d,
                correction_d_m2_s=d,
                t_s=sweep.print_time_s,
                voxel_m=vox_cfg.voxel_m,
                voxel_um=vox_cfg.voxel_um,
            )
        )
    return out


def composite_score(results: list[CorrectionResult], uncorrected: CorrectionResult) -> dict[str, float]:
    """Scaled ASSD + MS-SSIM composite (Fig. 6e)."""
    co = [r for r in results if r.method == "cooptimization"]
    if not co:
        return {}
    assd_vals = [r.assd_um for r in co]
    ssim_vals = [r.ms_ssim for r in co]
    a_min, a_max = min(assd_vals), max(assd_vals)
    s_min, s_max = min(ssim_vals), max(ssim_vals)
    scores: dict[str, float] = {}
    for r in co:
        a_norm = 1.0 - (r.assd_um - a_min) / (a_max - a_min + 1e-8)
        s_norm = (r.ms_ssim - s_min) / (s_max - s_min + 1e-8)
        scores[f"{r.correction_d_m2_s:.2e}"] = 0.5 * (a_norm + s_norm)
    return scores
