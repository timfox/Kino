"""Resolution-free geo-mapping framework card and demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.resfree_geo.beltrami import (
    beltrami_energy,
    beltrami_from_uv,
    maximal_dilatation,
    mild_sinusoidal_map,
    synthesize_beltrami_field,
)
from ltx_trainer.resfree_geo.config import ResfreeGeoConfig
from ltx_trainer.resfree_geo.dem import fick_velocity, jacobian_det_2d, ring_test_density, sin_cos_test_density, smooth_l1
from ltx_trainer.resfree_geo.encoding import encode_field, multi_resolution_stack, random_smooth_field
from ltx_trainer.resfree_geo.layout import LIMITATIONS
from ltx_trainer.resfree_geo.paper_tables import (
    framework_comparison,
    table2_beltrami_reconstruction,
    table3_resolution_beltrami,
    table4_deq_square_boundary,
    table5_deq_free_boundary,
    table6_deq_resolution,
    table7_dem3d_cube_boundary,
    table8_dem3d_free_boundary,
    table9_dem3d_resolution,
)
from ltx_trainer.resfree_geo.refinement import line_search_alpha, refined_mapping
from ltx_trainer.resfree_geo.surrogate import TinyGeoSurrogate, forward_mapping, square_boundary_mask


def framework_card(cfg: ResfreeGeoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ResfreeGeoConfig()
    return {
        "name": "Resolution-free Neural Geometric Mapping Surrogate",
        "paper": cfg.paper_arxiv,
        "authors": cfg.authors,
        "idea": (
            "Learn G: p(·) → u(·) without fixed grids: coordinate-augmented multi-resolution encoding, "
            "Enhanced U-Net with edge-aware conv, data-free losses (Beltrami, DEM/DEQ), "
            "and optional α stretching refinement."
        ),
        "architecture": cfg.architecture,
        "training_mode": cfg.training_mode,
        "problems": list(cfg.problems),
        "baselines": list(cfg.baselines),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "framework_comparison": framework_comparison(),
        "table2_beltrami": table2_beltrami_reconstruction(),
        "table3_resolution": table3_resolution_beltrami(),
        "table4_deq_square": table4_deq_square_boundary(),
        "table5_deq_free": table5_deq_free_boundary(),
        "table6_deq_resolution": table6_deq_resolution(),
        "table7_dem3d_cube": table7_dem3d_cube_boundary(),
        "table8_dem3d_free": table8_dem3d_free_boundary(),
        "table9_dem3d_resolution": table9_dem3d_resolution(),
    }


def evaluation_demo(cfg: ResfreeGeoConfig | None = None, *, seed: int = 7) -> dict[str, Any]:
    cfg = cfg or ResfreeGeoConfig()
    torch.manual_seed(seed)
    h, w = 32, 32

    p = random_smooth_field(h, w)
    xs = torch.linspace(0, 1, w)
    ys = torch.linspace(0, 1, h)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0)
    encoded = encode_field(coords, p.unsqueeze(0).unsqueeze(0))
    multi = multi_resolution_stack(encoded)

    model = TinyGeoSurrogate(multi.shape[1], 2)
    mask = square_boundary_mask(h, w)
    u, disp = forward_mapping(model, multi, mask)

    u_map, v_map = mild_sinusoidal_map(0.06, 0.04, n=h)
    mu_gt = synthesize_beltrami_field(h, w)
    mu_from_map = beltrami_from_uv(u_map, v_map)
    energy = beltrami_energy(mu_from_map)
    k_max = float(maximal_dilatation(mu_from_map).max().item())

    rho = sin_cos_test_density(h, w)
    vx, vy = fick_velocity(rho)
    jac = jacobian_det_2d(u_map, v_map)

    def obj(mapped: Tensor) -> torch.Tensor:
        return (mapped - torch.stack([u_map, v_map], dim=0)).pow(2).mean()

    alpha = line_search_alpha(disp, mask, torch.stack([xx, yy], dim=0), obj, steps=6)
    u_ref = refined_mapping(torch.stack([xx, yy], dim=0), disp, mask, alpha)

    t2 = table2_beltrami_reconstruction()
    t4 = table4_deq_square_boundary()

    return {
        "encoded_channels": int(encoded.shape[1]),
        "multi_res_channels": int(multi.shape[1]),
        "mapping_shape": list(u.shape),
        "beltrami_energy": float(energy.item()),
        "max_dilatation": k_max,
        "mean_mu_map": float(mu_from_map.mean().item()),
        "fick_v_norm": float((vx.pow(2) + vy.pow(2)).mean().sqrt().item()),
        "min_jacobian_gt": float(jac.min().item()),
        "alpha_refine": alpha.tolist(),
        "refine_loss": float(obj(u_ref).item()),
        "dem_smooth_l1": float(smooth_l1(rho, ring_test_density(h, w)).item()),
        "table2_case_a_delta_mu": t2[0]["mean_delta_mu"],
        "table4_ring_std_map": t4[3]["std_rho_map"],
    }
