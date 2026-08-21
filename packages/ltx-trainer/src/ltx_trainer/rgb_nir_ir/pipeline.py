"""RGB–NIR IR framework card and evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.rgb_nir_ir.brdf import disney_brdf_nir
from ltx_trainer.rgb_nir_ir.capture import exposure_schedule_ms, hdr_bracket_mean, isolate_nir_flash
from ltx_trainer.rgb_nir_ir.config import RgbNirIrConfig
from ltx_trainer.rgb_nir_ir.layout import LIMITATIONS
from ltx_trainer.rgb_nir_ir.paper_tables import table_i_inverse_rendering
from ltx_trainer.rgb_nir_ir.stages import (
    monte_carlo_rgb_pixel_smoke,
    stage1_geometry_init_loss,
    stage2_basis_optimization_smoke,
    stage2_nir_flash_loss,
    stage3_rgb_environment_loss,
)


def framework_card(cfg: RgbNirIrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RgbNirIrConfig()
    return {
        "name": "Ambient-robust RGB–NIR Inverse Rendering",
        "paper": cfg.paper_arxiv,
        "venue": cfg.venue,
        "doi": cfg.doi,
        "lab": cfg.lab,
        "idea": (
            "Active NIR flash (imperceptible) yields ambient-invariant point-light shading; "
            "ambient RGB supplies visible reflectance. Three stages: 2DGS geometry init, "
            "NIR basis-BRDF + geometry refine, RGB albedo + env map with shared σ/m."
        ),
        "imaging_system": {
            "camera": cfg.camera,
            "nir_flash": cfg.nir_flash,
            "robot": list(cfg.robot),
        },
        "stages": list(cfg.stages),
        "baselines": list(cfg.baselines),
        "dataset": {
            "real_objects": cfg.real_objects,
            "real_environments": cfg.real_environments,
            "synthetic_objects": cfg.synthetic_objects,
            "views_per_scene_min": cfg.views_per_scene_min,
        },
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle(cfg: RgbNirIrConfig | None = None) -> dict[str, Any]:
    return {"table_i": table_i_inverse_rendering()}


def evaluation_demo(cfg: RgbNirIrConfig | None = None, *, seed: int = 7) -> dict[str, Any]:
    cfg = cfg or RgbNirIrConfig()
    torch.manual_seed(seed)

    # Capture pipeline
    nir_on = torch.rand(32, 32) * 0.5 + 0.3
    nir_off = torch.rand(32, 32) * 0.2
    nir_flash = isolate_nir_flash(nir_on, nir_off)
    hdr_rgb = hdr_bracket_mean([torch.rand(3, 16, 16) * 0.4 for _ in range(3)])

    # Stage 1
    rgb_gt = torch.rand(3, 16, 16)
    l1 = stage1_geometry_init_loss(rgb_gt * 0.9, rgb_gt)

    # Stage 2
    nir_gt = nir_flash
    nir_pred = nir_flash * 0.95 + 0.02
    s2 = stage2_nir_flash_loss(nir_pred, nir_gt)
    basis = stage2_basis_optimization_smoke(nir_flash, num_bases=cfg.nir_basis_count)

    # BRDF spot check
    n_i = torch.tensor(0.8)
    n_o = torch.tensor(0.7)
    n_h = torch.tensor(0.85)
    brdf_val = disney_brdf_nir(n_i, n_o, n_h, rho_nir=torch.tensor(0.5), roughness=torch.tensor(0.3), metallic=torch.tensor(0.1))

    # Stage 3
    rgb_pred = rgb_gt * 0.92
    grad_rgb = torch.randn(16, 16)
    grad_nir = torch.randn(16, 16) * 0.1
    s3 = stage3_rgb_environment_loss(rgb_pred, rgb_gt, grad_rgb, grad_nir)
    mc = monte_carlo_rgb_pixel_smoke(torch.tensor(0.6), torch.tensor(1.0))

    table = table_i_inverse_rendering()
    ours = next(r for r in table if r["method"] == "Ours")
    wild = next(r for r in table if r["method"] == "WildLight")

    return {
        "nir_flash_mean": float(nir_flash.mean().item()),
        "hdr_rgb_shape": list(hdr_rgb.shape),
        "exposure_schedule_ms": exposure_schedule_ms(),
        "stage1_l1": float(l1.item()),
        "stage2_total": float(s2["total"].item()),
        "stage2_basis": basis,
        "brdf_spot": float(brdf_val.item()),
        "stage3_total": float(s3["total"].item()),
        "mc_rgb": float(mc.item()),
        "paper_albedo_psnr": ours["albedo_psnr"],
        "wildlight_albedo_psnr": wild["albedo_psnr"],
        "psnr_gain_vs_wildlight": ours["albedo_psnr"] - wild["albedo_psnr"],
    }
