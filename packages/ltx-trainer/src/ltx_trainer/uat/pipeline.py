"""Framework card, Tables 1–4, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.uat.config import UATConfig
from ltx_trainer.uat.diffusion import apply_random_mask, noisy_audio_latent, text_loss_weight
from ltx_trainer.uat.model import DualStreamDiTStub, masked_ce, velocity_mse


def framework_card(cfg: UATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UATConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "demo_url": cfg.demo_url,
        "params_b": cfg.params_b,
        "architecture": {
            "paradigm": "diffusion-centric (continuous audio + masked discrete text)",
            "backbone": cfg.backbone,
            "dit_blocks": cfg.dit_blocks,
            "hidden_dim": cfg.hidden_dim,
            "audio_vae": cfg.audio_vae,
            "text_encoder": cfg.text_encoder,
        },
        "tasks": ["text-to-audio generation", "text-guided audio editing (SDEdit)", "audio captioning"],
        "training": {
            "samples": cfg.training_samples,
            "hours": cfg.training_hours,
            "lambda_text": cfg.lambda_text,
        },
        "inference": {
            "steps": cfg.inference_steps,
            "cfg_scale": cfg.cfg_scale,
            "edit_start_step": cfg.edit_start_step,
        },
    }


def table1_tta_generation() -> dict[str, Any]:
    """Table 1 — UAT (Ours) on AudioCaps and VGGSound test sets."""
    return {
        "AudioCaps": {
            "KL": 1.39,
            "IS": 12.47,
            "FD": 14.47,
            "FAD": 2.87,
            "CLAP": 0.491,
        },
        "VGGSound": {
            "KL": 1.28,
            "IS": 9.34,
            "FD": 22.07,
            "FAD": 4.91,
            "CLAP": 0.434,
        },
    }


def table2_human_eval() -> dict[str, float]:
    """Table 2 — OVL and REL (1–5 scale)."""
    return {"OVL": 4.260, "REL": 4.260, "ground_truth_OVL": 4.347, "ground_truth_REL": 4.407}


def table3_audio_editing() -> dict[str, dict[str, float]]:
    """Table 3 — Add / Delete / Replace (Ours)."""
    return {
        "Add": {"CLAP": 0.406, "FAD": 3.220, "IS": 4.072},
        "Delete": {"CLAP": 0.350, "FAD": 4.243, "IS": 3.325},
        "Replace": {"CLAP": 0.439, "FAD": 5.199, "IS": 3.682},
    }


def table4_captioning() -> dict[str, float]:
    """Table 4 — audio captioning (Ours, 1.7B)."""
    return {
        "CIDEr": 0.406,
        "SPICE": 0.139,
        "SPIDEr": 0.272,
        "SBERT-SIM": 0.572,
        "FENSE": 54.08,
        "params_b": 1.7,
    }


def forward_smoke(cfg: UATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UATConfig()
    model = DualStreamDiTStub(cfg)
    b = 2
    z0 = torch.randn(b, cfg.demo_latent_len)
    t = torch.rand(b)
    zt, v_target, _ = noisy_audio_latent(z0, t)
    text = torch.randint(0, cfg.demo_vocab, (b, cfg.demo_text_len))
    tau = torch.rand(b)
    corrupted, mask = apply_random_mask(text, tau, mask_id=0)
    v_pred, logits = model(zt, corrupted, t, text_mask=mask)
    la = float(velocity_mse(v_pred, v_target).detach())
    lt = float(masked_ce(logits, text, mask).detach())
    w = float(text_loss_weight(tau).mean().detach())
    return {
        "audio_loss": la,
        "text_loss": lt,
        "joint_loss": la + cfg.lambda_text * lt,
        "text_weight_mean": w,
        "latent_shape": list(z0.shape),
        "logits_shape": list(logits.shape),
        "dit_stub_blocks": len(model.blocks),
    }


def evaluation_demo(cfg: UATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UATConfig()
    return {
        "framework": framework_card(cfg),
        "table1_tta": table1_tta_generation(),
        "table2_human": table2_human_eval(),
        "table3_editing": table3_audio_editing(),
        "table4_captioning": table4_captioning(),
        "forward": forward_smoke(cfg),
    }
