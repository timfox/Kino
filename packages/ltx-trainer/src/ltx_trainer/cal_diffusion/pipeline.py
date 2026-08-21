"""Framework card, demos, smoke for CAL diffusion co-optimization."""

from __future__ import annotations

from typing import Any

from ltx_trainer.cal_diffusion.config import CALDiffusionConfig, DiffusionSweep, VoxelConfig
from ltx_trainer.cal_diffusion.identification import best_identified_d, identify_diffusivity
from ltx_trainer.cal_diffusion.metrics import (
    operating_guidelines,
    table_corrected_fig6,
    table_deconv_fig7,
    table_uncorrected_fig3,
)
from ltx_trainer.cal_diffusion.simulation import (
    composite_score,
    correction_sweep,
    run_cooptimization,
    run_rl_deconvolution,
    run_uncorrected,
)
from ltx_trainer.cal_diffusion.tomography import plate_target_volume


def framework_card(cfg: CALDiffusionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CALDiffusionConfig()
    return {
        "name": "CAL-Diffusion",
        "paper": cfg.paper_note,
        "title": "Co-optimization of diffusive and tomographic blur in CAL",
        "method": "BCLP projection optimization with modified Radon + diffusion kernel (Eq. 5–8)",
        "kernel": "Steady-state diffusion around cubic sink; single D from uncorrected prints",
        "metrics": ["ASSD", "MS-SSIM", "thickness maps"],
        "material": "PETA + CQ/EDAB + TEMPO; 47 μm voxels; 6 mm vial",
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy FFT stub — not CAL-software-Matlab / ASTRA Radon pipeline.",
        "Gaussian proxy for Eq. (1) integral kernel; no absorption map (S.5) by default.",
        "Micro-CT registration and shrinkage are not replayed — synthetic plate geometry only.",
        "BCLP uses fixed iteration count, not full BB spectral step schedule (S.7).",
        "Fig. 7 deconv ASSD/SSIM are literature anchors for sanity checks on stub ordering.",
    ]


def evaluation_demo(cfg: CALDiffusionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CALDiffusionConfig()
    sweep = DiffusionSweep()
    vox = VoxelConfig()
    target = plate_target_volume(vox.volume_shape)

    id_results = identify_diffusivity(sweep, vox)
    corr = correction_sweep(sweep, vox)
    unc = next(r for r in corr if r.method == "uncorrected")
    co_notional = next(r for r in corr if r.method == "cooptimization" and r.correction_d_m2_s == sweep.notional_d_m2_s)
    rl = run_rl_deconvolution(
        target,
        true_d_m2_s=sweep.notional_d_m2_s,
        correction_d_m2_s=sweep.notional_d_m2_s,
        t_s=sweep.print_time_s,
        voxel_m=vox.voxel_m,
        voxel_um=vox.voxel_um,
    )
    comp = composite_score(corr, unc)

    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "identification": {
            "best_d_m2_s": best_identified_d(id_results),
            "sweep": [r.__dict__ for r in id_results],
        },
        "correction": {
            "uncorrected": unc.__dict__,
            "cooptimization_notional": co_notional.__dict__,
            "rl_deconvolution": rl.__dict__,
            "composite_by_d": comp,
        },
        "tables": {
            "fig3": table_uncorrected_fig3(),
            "fig6": table_corrected_fig6(),
            "fig7": table_deconv_fig7(),
        },
    }


def evaluation_smoke(cfg: CALDiffusionConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    sweep = DiffusionSweep()
    unc = demo["correction"]["uncorrected"]
    co = demo["correction"]["cooptimization_notional"]
    rl = demo["correction"]["rl_deconvolution"]
    fig7 = demo["tables"]["fig7"]
    best_d = demo["identification"]["best_d_m2_s"]

    assert abs(best_d - sweep.notional_d_m2_s) / sweep.notional_d_m2_s < 0.6
    assert co["assd_um"] < unc["assd_um"]
    assert co["ms_ssim"] > unc["ms_ssim"]
    assert co["assd_um"] < rl["assd_um"]
    assert co["bclp_loss"] < rl["bclp_loss"]
    assert rl["ms_ssim"] <= fig7["experimental_ms_ssim"] + 0.15

    comp = demo["correction"]["composite_by_d"]
    peak_d = max(comp, key=comp.get)
    peak_val = float(peak_d)
    assert 0.5e-10 <= peak_val <= 6.0e-10
    assert comp[peak_d] >= 0.45

    return {
        "status": "ok",
        "paper": (cfg or CALDiffusionConfig()).paper_note,
        "best_identified_d": best_d,
        "co_assd_um": co["assd_um"],
        "demo_keys": list(demo.keys()),
    }
