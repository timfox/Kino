"""Framework card, demos, benchmark manifest (arXiv:2605.29569)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.lora_key.config import LoRAKeyConfig
from ltx_trainer.lora_key.gop import cosine_similarity_flat, gradient_orthogonal_projection
from ltx_trainer.lora_key.losses import semantic_cosine_loss, watermark_consistency_loss
from ltx_trainer.lora_key.ltx_bridge import LoRAKeyLTXBridge, ltx_integration_notes
from ltx_trainer.lora_key.metrics import (
    gop_cosine_ablation,
    table_i_fidelity_watermark,
    table_iii_scalability,
    table_iv_cross_architecture,
)
from ltx_trainer.lora_key.prior import LatentWatermarkPrior, prior_loss
from ltx_trainer.lora_key.verification import verify_ownership


def framework_card(cfg: LoRAKeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoRAKeyConfig()
    return {
        "name": "LoRA-Key",
        "paper": cfg.paper_arxiv,
        "paradigm": "user_centric_reusable_watermark_lora",
        "stages": [
            "latent watermark prior (frozen VAE, E_ψ / D_φ)",
            "semantic-orthogonal Watermark LoRA + GOP",
            "training-free linear superposition with style LoRA",
            "ownership verification (binomial hypothesis test)",
        ],
        "message_bits": cfg.message_bits,
        "base_model": cfg.base_model,
        "vs_authenlora": "one-time key vs per-LoRA retraining",
    }


def paper_limitations() -> list[str]:
    return [
        "Evaluated primarily on SD-family T2I; LTX AV LoRA path is integration guidance only.",
        "Application-layer payload semantics not watermarked.",
        "Strong distortions or rank-mismatched merges may reduce bit accuracy.",
    ]


def benchmark_manifest(cfg: LoRAKeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoRAKeyConfig()
    return {
        "datasets": ["StyleDrop", "DreamBooth", "Civitai/HF community LoRAs"],
        "metrics": ["FID", "CLIP", "DreamSim", "Bit Acc", "TPR @ FPR=1e-6"],
        "baselines": ["DwtDct", "RivaGAN", "Stable Signature", "AquaLoRA", "AuthenLoRA"],
        "message_bits": cfg.message_bits,
    }


def evaluation_demo(*, cfg: LoRAKeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoRAKeyConfig()
    torch.manual_seed(3)
    L = cfg.message_bits
    msg = torch.randint(0, 2, (L,))
    prior = LatentWatermarkPrior(message_bits=L, latent_dim=256)
    z = torch.randn(1, 256)
    z_wm = prior.embed(z, msg)
    decoded = prior.decode_message(z_wm)[0]
    pl = prior_loss(msg, decoded)

    g_wm = torch.randn(128)
    g_sem = torch.randn(128)
    g_proj = gradient_orthogonal_projection(g_wm, g_sem, eps=cfg.gop_eps)
    cos_before = cosine_similarity_flat(g_wm, g_sem)
    cos_after = cosine_similarity_flat(g_proj, g_sem)

    eps_b = torch.randn(4, 8)
    l_wm = float(watermark_consistency_loss(eps_b + 0.01, eps_b).item())
    ref_f = torch.randn(2, cfg.dino_feature_dim)
    wm_f = ref_f + torch.randn_like(ref_f) * 0.05
    l_sem = float(semantic_cosine_loss(ref_f, wm_f).item())

    bridge = LoRAKeyLTXBridge(cfg, d_in=32, d_out=32)
    x = torch.randn(2, 32)
    y = bridge(x)

    recovered = (decoded > 0.5).float()
    verdict = verify_ownership(msg, recovered, target_fpr=cfg.target_fpr)

    ours_content = next(r for r in table_i_fidelity_watermark() if r["method"] == "Ours" and r["task"] == "Content")
    scale = next(r for r in table_iii_scalability() if r["method"] == "LoRA-Key" and r["loras"] == 10)

    return {
        "prior_loss": {k: round(v, 4) for k, v in pl.items()},
        "gop_cosine_before": round(cos_before, 3),
        "gop_cosine_after": round(cos_after, 3),
        "watermark_consistency_mse": round(l_wm, 6),
        "semantic_loss": round(l_sem, 4),
        "ltx_bridge_output_shape": list(y.shape),
        "verification": verdict,
        "paper_content_bit_acc_adv": ours_content["bit_acc_adv"],
        "paper_scale_avg_h_per_lora": scale["avg_per_lora_h"],
        "gop_ablation": gop_cosine_ablation(),
        "cross_arch": table_iv_cross_architecture(),
        "conclusion": "Reusable Watermark LoRA + GOP preserves fidelity and robust verification under composition.",
    }


def training_step_demo(*, cfg: LoRAKeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoRAKeyConfig()
    g_wm = torch.randn(64)
    g_sem = torch.randn(64)
    g_proj = gradient_orthogonal_projection(g_wm, g_sem)
    return {
        "g_proj_norm": round(float(g_proj.norm().item()), 4),
        "merged_rank": cfg.watermark_lora_rank,
        "deploy_formula": "Theta0 + alpha*DeltaTheta_style + gamma*DeltaTheta_key",
        "ltx_notes": ltx_integration_notes(cfg),
    }
