"""CLEAR framework card, benchmark tables, and smoke demos."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.clear.config import CLEARConfig
from ltx_trainer.clear.gumbel import gumbel_softmax, select_layer_index
from ltx_trainer.clear.losses import lcon_separability, lsae_reconstruction
from ltx_trainer.clear.sae import (
    SparseAutoencoder,
    aggregate_activation,
    concept_erasure,
    specificity_mask,
)


def framework_card(cfg: CLEARConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CLEARConfig()
    return {
        "name": "CLEAR",
        "paper": "arXiv:2605.25941",
        "title": "Concept–Layer Alignment for Text-to-Video Concept Erasure",
        "full_name": "Concept-Layer Erasure Alignment Framework",
        "stage_i": "Gumbel-Softmax depth search + SAE separability (L_con)",
        "stage_ii": "single-layer sparse feature subtraction at inference",
        "models": list(cfg.backbone_models),
        "text_encoder": cfg.text_encoder,
        "training_free_at_inference": True,
        "sae_hidden_dim": cfg.d_sae,
        "default_gamma": cfg.intervention_gamma,
    }


def table_wan_objects() -> dict[str, dict[str, float]]:
    """Table 1a — Wan2.2-5B object erasure (generative rate %)."""
    return {
        "origin": {"avg_gen_rate": 61.8, "img_q": 0.6910, "aes_q": 0.5981},
        "negprompt": {"avg_gen_rate": 54.3, "img_q": 0.6644, "aes_q": 0.5882},
        "safree": {"avg_gen_rate": 28.1, "img_q": 0.6304, "aes_q": 0.5587},
        "t2vunlearning": {"avg_gen_rate": 24.5, "img_q": 0.6652, "aes_q": 0.5197},
        "clear": {
            "avg_gen_rate": 12.8,
            "french_horn": 7.8,
            "parachute": 18.4,
            "img_q": 0.7025,
            "aes_q": 0.5758,
            "overall_consistency": 0.1896,
        },
    }


def table_cogvideo_objects() -> dict[str, dict[str, float]]:
    """Table 1b — CogVideoX-2B object erasure."""
    return {
        "origin": {"avg_gen_rate": 56.2, "img_q": 0.5505},
        "t2vunlearning": {"avg_gen_rate": 7.4, "img_q": 0.3907},
        "clear": {"avg_gen_rate": 7.1, "img_q": 0.4683, "aes_q": 0.4484},
    }


def table_nudity_wan() -> dict[str, dict[str, float]]:
    """Table 2a — nudity on Wan2.2-5B."""
    return {
        "origin": {"gen_rate": 67.3, "img_q": 0.6913},
        "t2vunlearning": {"gen_rate": 18.6, "img_q": 0.6737},
        "clear": {"gen_rate": 11.1, "img_q": 0.6928, "aes_q": 0.5546},
    }


def table_ablation_layer_search() -> dict[str, dict[str, float | int | str]]:
    """Table 5 — layer search ablation."""
    return {
        "springer_dog_manual": {"block": 12, "gen_rate": 0.0, "cost": "high"},
        "springer_dog_wo_lcon": {"block": 15, "gen_rate": 14.0, "cost": "low"},
        "springer_dog_clear": {"block": 2, "gen_rate": 4.8, "cost": "low"},
        "parachute_clear": {"block": 6, "gen_rate": 18.4, "cost": "low"},
        "nudity_clear": {"block": 18, "gen_rate": 10.9, "cost": "low"},
    }


def table_gamma_ablation() -> dict[str, float]:
    """Table 6 — intervention strength γ on nudity."""
    return {8.0: 28.4, 10.0: 11.1, 12.0: 3.2}


def training_step_demo(cfg: CLEARConfig | None = None) -> dict[str, float]:
    """Smoke: Gumbel layer pick, SAE recon, L_con, erasure step."""
    cfg = cfg or CLEARConfig()
    torch.manual_seed(0)
    d_model = cfg.demo_d_model
    d_sae = cfg.demo_d_sae

    alpha = torch.zeros(cfg.num_layers)
    alpha[cfg.selected_blocks["parachute"]] = 2.0
    p = gumbel_softmax(alpha, tau=cfg.gumbel_tau_max)
    l_star = select_layer_index(p)

    sae = SparseAutoencoder(d_model, d_sae)
    h_pos = torch.randn(d_model)
    h_neg = torch.randn(d_model)
    f_pos, h_hat_pos = sae(h_pos)
    f_neg = sae.encode(h_neg)
    m_spec = specificity_mask(f_neg)
    loss_rec = float(lsae_reconstruction(h_pos, h_hat_pos, f_pos, sparsity_lambda=cfg.sparsity_lambda))
    s_spe = aggregate_activation(f_pos, m_spec)
    s_uni = aggregate_activation(f_neg, 1.0 - m_spec)
    loss_con = float(lcon_separability(s_spe, s_uni, eps=cfg.stability_eps))

    h_erased = concept_erasure(h_pos, sae, m_spec, gamma=cfg.intervention_gamma)
    delta = float(torch.norm(h_erased - h_pos))

    return {
        "selected_layer": float(l_star),
        "layer_prob_max": float(p.max()),
        "loss_rec": loss_rec,
        "loss_con": loss_con,
        "erasure_delta_norm": delta,
        "s_spe": float(s_spe),
        "s_uni": float(s_uni),
    }


def evaluation_demo() -> dict[str, Any]:
    wan = table_wan_objects()
    cog = table_cogvideo_objects()
    nud = table_nudity_wan()
    abl = table_ablation_layer_search()
    step = training_step_demo()

    return {
        **step,
        "wan_clear_avg_gen": wan["clear"]["avg_gen_rate"],
        "wan_origin_avg_gen": wan["origin"]["avg_gen_rate"],
        "french_horn_clear": wan["clear"]["french_horn"],
        "cog_clear_avg_gen": cog["clear"]["avg_gen_rate"],
        "nudity_clear_gen": nud["clear"]["gen_rate"],
        "nudity_origin_gen": nud["origin"]["gen_rate"],
        "clear_img_q_vs_origin": wan["clear"]["img_q"] - wan["origin"]["img_q"],
        "parachute_block": float(abl["parachute_clear"]["block"]),
        "gamma_default_gen_rate": table_gamma_ablation()[10.0],
    }
