"""FPD framework card, benchmark tables, and training demo."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.fpd.config import DRIFT_BANDWIDTHS, FPDConfig
from ltx_trainer.fpd.drift import drift_loss
from ltx_trainer.fpd.masking import forward_mask_tokens, remask_fraction
from ltx_trainer.fpd.ste import sample_hard_indices, straight_through_embedding


def framework_card(cfg: FPDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FPDConfig()
    return {
        "name": "FPD",
        "paper": "arXiv:2605.21484",
        "title": "One-Step Distillation of Discrete Diffusion via Fixed-Point Iteration",
        "teachers": ["MaskGIT", "MaskGen-L"],
        "backbone": "DINOv3 ViT-B/16",
        "inference_steps": 1,
        "aux_score_net": cfg.requires_aux_score_net,
        "losses": ["L_drift", "L_gan_optional"],
        "drift_bandwidths": list(cfg.drift_bandwidths),
        "backbone_blocks": list(cfg.backbone_blocks),
        "fixed_point": "p_θ = (T^r_φ ∘ M_r)_# p_θ",
    }


def table_geneval_text2image() -> dict[str, dict[str, float | int | bool]]:
    """Table 1 — GenEval (selected rows)."""
    return {
        "maskgen_teacher_16": {"steps": 16, "params_b": 0.6, "aux_score_net": False, "overall": 0.48},
        "maskgen_dimo_1": {"steps": 1, "params_b": 0.6, "aux_score_net": True, "overall": 0.42},
        "maskgen_fpd_1": {"steps": 1, "params_b": 0.6, "aux_score_net": False, "overall": 0.45},
        "sdxl_dmd2_1": {"steps": 1, "params_b": 2.6, "aux_score_net": True, "overall": 0.55},
    }


def table_imagenet_class_cond() -> dict[str, dict[str, float | int | bool]]:
    """Table 2 — ImageNet-256 with MaskGIT teacher."""
    return {
        "maskgit_teacher_16": {"steps": 16, "aux_score_net": False, "fid": 6.60, "is": 224},
        "maskgit_teacher_1": {"steps": 1, "aux_score_net": False, "fid": 91.35, "is": 13},
        "sdtt_1": {"steps": 1, "aux_score_net": False, "fid": 90.40, "is": 14},
        "dimo_1": {"steps": 1, "aux_score_net": True, "fid": 6.91, "is": 214},
        "fpd_1": {"steps": 1, "aux_score_net": False, "fid": 6.90, "is": 215},
    }


def table_ablation_routing() -> dict[str, float]:
    """Table 3a — gradient routing and GAN."""
    return {
        "soft_emb": 0.38,
        "soft_emb_gan": 0.42,
        "ste": 0.43,
        "ste_gan": 0.45,
    }


def table_ablation_refinement_source() -> dict[str, float]:
    """Table 3b — teacher refinement source."""
    return {"random": 0.22, "student": 0.43}


def table_ablation_drift_space() -> dict[str, dict[str, float]]:
    """Table 4 — drift space and backbone ablations."""
    return {
        "pixel_multi_bw": {"overall": 0.21},
        "feature_single_bw": {"overall": 0.41},
        "feature_multi_bw": {"overall": 0.43},
        "blocks_2_5_8_11_grid": {"overall": 0.43},
        "blocks_8_11_grid": {"overall": 0.42},
    }


def demo_codebook(num_codes: int = 64, dim: int = 16) -> Tensor:
    torch.manual_seed(0)
    return torch.randn(num_codes, dim)


def training_step_demo(cfg: FPDConfig | None = None) -> dict[str, float]:
    """Smoke: student draft → re-mask → STE → drift loss."""
    cfg = cfg or FPDConfig()
    k, d, l = 32, 16, 64
    codebook = demo_codebook(k, d)
    mask_id = k - 1
    z_init = torch.randint(0, k - 1, (l,))
    z_init = forward_mask_tokens(z_init, cfg.r_init, mask_id=mask_id)
    logits = torch.randn(l, k)
    hard = sample_hard_indices(logits)
    emb = straight_through_embedding(logits, hard, codebook)
    z_draft = hard
    z_refined = remask_fraction(z_draft, mask_id, r=0.3)
    # Synthetic lifted features
    x_s = emb.mean(dim=0, keepdim=True).expand(4, -1)
    x_t = x_s + 0.1 * torch.randn_like(x_s)
    l_drift = drift_loss(x_s, x_t, cfg.drift_bandwidths)
    l_gan = torch.tensor(0.0)
    l_total = l_drift + cfg.lambda_gan * l_gan
    return {
        "l_drift": float(l_drift.item()),
        "l_gan": float(l_gan.item()),
        "l_total": float(l_total.item()),
        "draft_len": float(z_draft.numel()),
        "refined_masked": float((z_refined == mask_id).sum().item()),
    }
