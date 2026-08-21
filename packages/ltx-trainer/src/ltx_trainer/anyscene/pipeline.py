"""AnyScene framework card and reference experiment tables."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.anyscene.bev import BEVLayoutEncoder
from ltx_trainer.anyscene.config import AnySceneConfig
from ltx_trainer.anyscene.ggve import (
    GGVEControlHints,
    render_coordinate_buffer,
    render_semantic_buffer,
    surround_expansion_schedule,
)
from ltx_trainer.anyscene.occupancy_vae import OccupancyVAE, vae_loss
from ltx_trainer.anyscene.stocc_dit import STOccDiT, flow_matching_loss, flow_matching_noisy_latent


def framework_card(cfg: AnySceneConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AnySceneConfig()
    return {
        "name": "AnyScene",
        "paper": "arXiv:2605.26113",
        "title": "Towards Highly Controllable Driving Scene Generation at Anywhere and Beyond",
        "factorization": "p(V|B) = p(V|O) · p(O|B)",
        "stages": {
            "occupancy": "BEV layout → STOccDiT semantic occupancy (nuCraftv2)",
            "video": "GGVE geometry-grounded multi-view expansion (Wan2.1 + VACE)",
        },
        "dataset": {
            "name": "nuCraftv2",
            "hz": cfg.dataset_hz,
            "bev": f"{cfg.bev_resolution}×{cfg.bev_resolution}, {cfg.bev_channels} channels @ {cfg.voxel_size_m}m",
            "occupancy": f"{cfg.bev_resolution}×{cfg.bev_resolution}×{cfg.occupancy_z}",
        },
        "stocc_dit": {
            "depth": cfg.dit_depth,
            "hidden": cfg.dit_hidden,
            "token_grid": cfg.token_grid,
            "conditioning": "BEV + occupancy token concat + causal temporal mask",
        },
        "ggve": {
            "backbone": "Wan2.1-T2V-14B",
            "control": "semantic + coordinate buffers + Plücker",
            "inference": "autoregressive surround expansion, arbitrary K views",
        },
    }


def table_occupancy_generation_nucraftv2() -> dict[str, float]:
    """Table 1 — occupancy generation on nuCraftv2-val."""
    return {
        "mIoU_3d": 19.01,
        "IoU_3d": 15.58,
        "mIoU_bev_topdown": 35.04,
        "IoU_bev_topdown": 60.53,
        "mIoU_bev_vs_layout": 52.45,
        "IoU_bev_vs_layout": 68.97,
        "FID_cam_render": 15.03,
        "KID_cam_render": 13.00,
        "FID_bev_render": 54.52,
        "KID_bev_render": 41.30,
    }


def table_occupancy_vae_reconstruction() -> dict[str, dict[str, float]]:
    """Table 2 — VAE reconstruction (selected rows)."""
    return {
        "x_scene_32x64x64": {"mIoU": 58.05, "IoU": 44.96},
        "x_scene_32x128x128": {"mIoU": 81.47, "IoU": 61.94},
        "uniscene_8x50x50": {"mIoU": 72.90, "IoU": 64.10},
        "ours_32x64x64": {"mIoU": 76.02, "IoU": 62.30},
        "ours_16x128x128": {"mIoU": 85.29, "IoU": 73.72},
        "ours_32x128x128": {"mIoU": 90.27, "IoU": 81.25},
    }


def table_video_generation() -> dict[str, dict[str, float | int | str]]:
    """Table 3 — FVD / mIoU / mAP (selected methods)."""
    return {
        "magicdrive_v2_16": {"frames": 16, "fvd": 94.84, "mIoU": 20.40, "mAP": 18.17},
        "uniscene_8": {"frames": 8, "fvd": 70.52, "mIoU": 21.75, "mAP": 10.32},
        "geniedrive_8": {"frames": 8, "fvd": 55.93, "mIoU": 30.97, "mAP": 19.17},
        "ours_8": {"frames": 8, "fvd": 35.98, "mIoU": 38.26, "mAP": 27.73},
        "ours_16": {"frames": 16, "fvd": 43.19, "mIoU": 38.26, "mAP": 27.73},
        "ours_49": {"frames": 49, "fvd": 57.57, "mIoU": 38.26, "mAP": 27.73},
    }


def table_ablation_stocc_dit() -> dict[str, dict[str, float]]:
    """Table 4 — STOccDiT design ablations."""
    return {
        "ours_tf_L24_concat": {"mIoU": 19.61, "IoU": 11.94},
        "w/o_teacher_forcing": {"mIoU": 3.23, "IoU": 0.98},
        "L16": {"mIoU": 12.22, "IoU": 10.11},
        "additive_bev": {"mIoU": 13.59, "IoU": 7.60},
    }


def table_ablation_ggve_buffers() -> dict[str, float]:
    """Table 5 — GGVE conditioning ablation (16 frames, FVD)."""
    return {
        "full": 43.19,
        "w/o_semantic": 47.05,
        "w/o_coordinate": 52.59,
        "w/o_both": 112.20,
    }


def nucraftv2_curation_stages() -> list[str]:
    """Supplement A — nuCraftv2 pipeline stages."""
    return [
        "cross_modal_3d_detection_20hz",
        "per_instance_sam3_sam3d_bank",
        "kiss_icp_pose_refinement",
        "dynamic_removal_panoptic_propagation",
        "shine_mapping_static_reconstruction",
        "gpu_voxelized_panoptic_assembly",
    ]


def evaluation_demo(*, device: str = "cpu") -> dict[str, Any]:
    """Smoke: VAE encode, STOccDiT flow step, GGVE hints, reference tables."""
    cfg = AnySceneConfig()
    dev = torch.device(device)
    b, h, w = 1, 32, 32
    z = cfg.occupancy_z
    labels = torch.randint(0, cfg.occupancy_classes, (b, h, w, z), device=dev)
    layout = (torch.rand(b, cfg.bev_channels, h, w, device=dev) > 0.9).float()

    vae = OccupancyVAE(cfg).to(dev)
    z_lat, mu, logvar, recon = vae(labels)
    loss_vae = float(vae_loss(recon, labels, mu, logvar, cfg).item())

    bev_enc = BEVLayoutEncoder(cfg).to(dev)
    bev_tok = bev_enc(layout)
    occ_tok = z_lat.flatten(2).transpose(1, 2)
    if occ_tok.shape[1] != bev_tok.shape[1]:
        occ_tok = torch.nn.functional.adaptive_avg_pool1d(
            occ_tok.transpose(1, 2), bev_tok.shape[1]
        ).transpose(1, 2)

    dit = STOccDiT(cfg, depth=2).to(dev)
    tau = torch.tensor([0.5], device=dev)
    noise = torch.randn_like(occ_tok)
    z_noisy = flow_matching_noisy_latent(occ_tok, tau, noise)
    pred_v = dit(z_noisy, bev_tok)
    loss_fm = float(flow_matching_loss(pred_v, occ_tok, noise, tau).item())

    sem = render_semantic_buffer(labels[0], 48, 80)
    coord = render_coordinate_buffer(labels[0], 48, 80)
    plk = torch.zeros(1, 48, 80, cfg.plucker_dim, device=dev)
    hints = GGVEControlHints(cfg).to(dev)
    hint = hints(sem, coord, plk)

    from ltx_trainer.anyscene.metrics import binary_iou, mean_iou

    pred_labels = labels.clone()
    return {
        "latent_shape": list(z_lat.shape),
        "vae_loss": loss_vae,
        "flow_matching_loss": loss_fm,
        "ggve_hint_shape": list(hint.shape),
        "binary_iou_random": binary_iou(pred_labels, labels),
        "mean_iou_random": mean_iou(pred_labels, labels, cfg.occupancy_classes),
        "surround_steps": len(surround_expansion_schedule()),
        "table1_mIoU_3d": table_occupancy_generation_nucraftv2()["mIoU_3d"],
        "table3_ours_fvd_16": table_video_generation()["ours_16"]["fvd"],
    }
