"""NEUROK framework card, demos, and benchmark bundles."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.neurok.active_subspace import active_subspace_basis, project_latent
from ltx_trainer.neurok.config import NeurokConfig
from ltx_trainer.neurok.kinematics import KinematicParameterization, configuration_manifold_dim
from ltx_trainer.neurok.lagrangian import lagrangian, simulate_latent_trajectory, total_energy
from ltx_trainer.neurok.layout import LIMITATIONS
from ltx_trainer.neurok.metrics import inverse_kinematics_smoke
from ltx_trainer.neurok.paper_tables import (
    table_i_inverse_kinematics,
    table_ii_generative_4d,
    user_study_headline,
)
from ltx_trainer.neurok.vae import toy_encoder_decoder_step


def framework_card(cfg: NeurokConfig | None = None) -> dict[str, Any]:
    cfg = cfg or NeurokConfig()
    k = cfg.num_latent_tokens * cfg.token_dim
    return {
        "name": "NEUROK",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "idea": (
            "Learn Neural Object Kinematics: instance-specific latent configuration space Z(M0) "
            "and decoder F mapping latents to plausible mesh deformations, then simulate dynamics "
            "with Euler–Lagrange equations in latent space (no category-specific physics priors)."
        ),
        "stages": [
            "Stage A: conditional VAE (E_cond, E_VAE, D) on large-scale 4D mesh trajectories",
            "Stage B: active-subspace reduction of latent dimension",
            "Stage C: Lagrangian latent simulation under forces / velocities / actions",
        ],
        "latent": {
            "tokens": cfg.num_latent_tokens,
            "token_dim": cfg.token_dim,
            "flat_dim": k,
            "reduced_dim": cfg.active_subspace_dim,
            "kl_weight": cfg.vae_kl_weight,
        },
        "training_data": "Objaverse-XL + PartNet-Mobility + synthetic physics sim (paper)",
        "baselines": list(cfg.baselines),
        "limitations": list(LIMITATIONS),
    }


def pipeline_demo(cfg: NeurokConfig | None = None, *, seed: int = 11) -> dict[str, Any]:
    cfg = cfg or NeurokConfig()
    torch.manual_seed(seed)
    k = cfg.num_latent_tokens * cfg.token_dim
    kq = cfg.active_subspace_dim

    vae = toy_encoder_decoder_step(
        num_samples=cfg.num_surface_samples,
        latent_tokens=cfg.num_latent_tokens,
        token_dim=cfg.token_dim,
        deform_dim=cfg.deformation_dim,
        kl_weight=cfg.vae_kl_weight,
    )

    decoder_w = torch.randn(cfg.num_surface_samples * cfg.deformation_dim, k) * 0.02
    basis = active_subspace_basis(decoder_w, reduced_dim=kq)
    z0 = torch.zeros(k)
    q0 = project_latent(z0, basis)
    assert q0.shape[0] == kq

    z_dot0 = torch.randn(k) * 0.01
    traj = simulate_latent_trajectory(
        z0,
        z_dot0,
        decoder_weight=decoder_w,
        steps=min(20, cfg.simulation_steps),
        dt=cfg.simulation_dt,
    )
    e0 = total_energy(traj[0], z_dot0)
    e1 = total_energy(traj[-1], z_dot0 * 0.95)

    ik = inverse_kinematics_smoke(latent_dim=kq, num_points=48, steps=30, seed=seed)
    n_verts = 100
    kin = KinematicParameterization(latent_dim=k, num_vertices=n_verts)

    return {
        "vae_losses": vae,
        "latent_dim": k,
        "reduced_dim": kq,
        "intrinsic_dof_estimate": configuration_manifold_dim(n_verts),
        "lagrangian_initial": round(lagrangian(traj[0], z_dot0).item(), 4),
        "trajectory_len": len(traj),
        "energy_first": round(e0, 4),
        "energy_last": round(e1, 4),
        "energy_drift": round(abs(e1 - e0), 4),
        "ik_smoke": ik,
        "kinematic_pair": {"Z_dim": kin.latent_dim, "vertex_dim": kin.num_vertices * 3},
    }


def evaluation_demo(cfg: NeurokConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    t1 = table_i_inverse_kinematics()
    ours = next(r for r in t1 if r["method"] == "NEUROK (ours)")
    demo["framework"] = framework_card(cfg)
    demo["paper_ik_chamfer_l1"] = ours["chamfer_l1"]
    demo["paper_ik_iou"] = ours["iou"]
    demo["user_study"] = user_study_headline()
    demo["limitations"] = LIMITATIONS
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    t1 = table_i_inverse_kinematics()
    t2 = table_ii_generative_4d()
    ours_ik = next(r for r in t1 if r["method"] == "NEUROK (ours)")
    kpd = next(r for r in t1 if r["method"] == "KeyPointDeformer")
    ours_4d = next(r for r in t2 if r["method"] == "NEUROK (ours)")
    phys = next(r for r in t2 if r["method"] == "PhysDreamer")
    return {
        "table_i_inverse_kinematics": t1,
        "table_ii_generative_4d": t2,
        "headline": {
            "ik_chamfer_l1_gain_vs_kpd": kpd["chamfer_l1"] - ours_ik["chamfer_l1"],
            "ik_iou": ours_ik["iou"],
            "user_alignment_pct": ours_4d["user_alignment_pct"],
            "user_realism_pct": ours_4d["user_realism_pct"],
            "vbench_aq_gain_vs_physdreamer": ours_4d["vbench_aq"] - phys["vbench_aq"],
            "worldscore_mm": ours_4d["worldscore_mm"],
        },
    }
