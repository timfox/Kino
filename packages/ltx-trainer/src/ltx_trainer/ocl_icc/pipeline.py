"""Framework card, demos, benchmark manifest (arXiv:2605.30211)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ocl_icc.analysis import (
    latent_distance,
    manifold_scatter_point,
    slot_diversity,
    slot_temporal_variance,
)
from ltx_trainer.ocl_icc.config import OCLICCConfig
from ltx_trainer.ocl_icc.losses import total_objective
from ltx_trainer.ocl_icc.ltx_bridge import OCLICCLTXBridge, ltx_integration_notes
from ltx_trainer.ocl_icc.metrics import (
    table1_object_discovery,
    table2_efficiency,
    table3_recognition,
    table4_ablation_movc,
    table5_collapse_ytvis,
)
from ltx_trainer.ocl_icc.streams import VideoOCLStreams


def framework_card(cfg: OCLICCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OCLICCConfig()
    icc_row = next(r for r in table1_object_discovery() if r["method"] == "RandSF.Q+ICC")
    ecc_row = next(r for r in table5_collapse_ytvis() if r["variant"] == "+ ECC")
    return {
        "name": "OCL ICC",
        "paper": cfg.paper_arxiv,
        "title": "Cycle Consistency in Video Object-Centric Learning",
        "method": "Implicit Cycle Consistency on reconstruction manifold (Eq. 8)",
        "vs_ecc": "ECC aligns latent slots (Eq. 4) — prone to collapse; ICC soft consensus on observations",
        "encoder_backbone": cfg.encoder_backbone,
        "num_slots": cfg.num_slots,
        "basis_methods": list(cfg.basis_methods),
        "datasets": list(cfg.datasets),
        "reported_movc_ari_icc": icc_row["MOVi-C_ARI"],
        "ecc_collapse_diversity": ecc_row["diversity"],
        "icc_diversity": next(r["diversity"] for r in table5_collapse_ytvis() if r["variant"] == "+ ICC"),
    }


def paper_limitations() -> list[str]:
    return [
        "Stub uses synthetic DINO-sized features; full MOVi/YTVIS pipelines are external.",
        "DINOv2-ViT-S/14 encoder weights not bundled.",
        "Recognition head (Table 3) and RandSF.Q basis training are reference tables only.",
        "LTX bridge provides multi-object slot QA smoke, not full segmentation export.",
    ]


def benchmark_manifest(cfg: OCLICCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OCLICCConfig()
    return {
        "datasets": list(cfg.datasets),
        "metrics": ["ARI", "ARI_fg", "mBO", "mIoU", "Top-1/3", "slot diversity", "temporal variance"],
        "baselines": ["VideoSAUR", "SlotContrast", "RandSF.Q", "SmoothSA", "+ECC", "+ICC"],
        "ablations": ["Hungarian ECC", "Non-chain recon", "ICC"],
    }


def _synthetic_clip(cfg: OCLICCConfig, *, t: int = 4, batch: int = 2) -> torch.Tensor:
    """(T, B, N, D) frozen-backbone feature clip."""
    n = cfg.spatial_h * cfg.spatial_w
    return torch.randn(t, batch, n, cfg.feature_dim)


def training_step_demo(cfg: OCLICCConfig | None = None) -> dict[str, float | str]:
    cfg = cfg or OCLICCConfig()
    torch.manual_seed(3)
    model = VideoOCLStreams(cfg)
    frames = _synthetic_clip(cfg)
    fw, bw = model.run_bidirectional(frames)
    targets = [frames[t].mean(dim=1) for t in range(frames.shape[0])]
    losses = total_objective(fw, bw, targets, mode="icc")
    return {
        "mode": "icc",
        "total_loss": float(losses["total"].detach().item()),
        "l_recon_fw": float(losses["l_recon_fw"].detach().item()),
        "l_icc": float(losses["l_icc"].detach().item()),
        "num_frames": float(frames.shape[0]),
        "num_slots": float(cfg.num_slots),
    }


def evaluation_demo(cfg: OCLICCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OCLICCConfig()
    torch.manual_seed(11)
    model = VideoOCLStreams(cfg)
    frames = _synthetic_clip(cfg, t=5)
    fw, bw = model.run_bidirectional(frames)
    targets = [frames[t].mean(dim=1) for t in range(frames.shape[0])]

    base = total_objective(fw, bw, targets, mode="baseline")
    ecc = total_objective(fw, bw, targets, mode="ecc")
    icc = total_objective(fw, bw, targets, mode="icc")
    hecc = total_objective(fw, bw, targets, mode="hungarian_ecc")

    div_fw = slot_diversity(fw.slots[0])
    var_fw = slot_temporal_variance(fw.slots)
    scatter = manifold_scatter_point(fw.slots, bw.slots, fw.reconstructions, bw.reconstructions, targets)

    bridge = OCLICCLTXBridge(cfg)
    bridge_out = bridge(frames)

    tab1 = table1_object_discovery()
    icc_best = max(tab1, key=lambda r: r["MOVi-C_ARI"])
    tab4 = table4_ablation_movc()
    tab5 = table5_collapse_ytvis()

    return {
        **training_step_demo(cfg),
        "loss_baseline": float(base["total"].detach().item()),
        "loss_ecc": float(ecc["total"].detach().item()),
        "loss_icc": float(icc["total"].detach().item()),
        "loss_hungarian_ecc": float(hecc["total"].detach().item()),
        "slot_diversity": round(div_fw, 4),
        "slot_temporal_variance": round(var_fw, 4),
        "latent_distance_fw_bw": round(latent_distance(fw.slots, bw.slots), 4),
        "manifold_scatter": {k: round(v, 4) for k, v in scatter.items()},
        "ltx_bridge_slots_shape": list(bridge_out["slots"].shape),
        "paper_best_movc_ari_method": icc_best["method"],
        "paper_best_movc_ari": icc_best["MOVi-C_ARI"],
        "paper_icc_ablation_ari": next(r["ARI"] for r in tab4 if r["variant"] == "+ ICC"),
        "paper_ecc_diversity_collapse": next(r["diversity"] for r in tab5 if r["variant"] == "+ ECC"),
        "paper_icc_diversity": next(r["diversity"] for r in tab5 if r["variant"] == "+ ICC"),
        "recognition_top1_icc": next(r["top1"] for r in table3_recognition() if "ICC" in str(r["method"])),
        "ltx_notes": ltx_integration_notes(cfg),
        "conclusion": "ICC aligns forward/backward on reconstruction manifold, avoiding ECC slot collapse.",
    }
