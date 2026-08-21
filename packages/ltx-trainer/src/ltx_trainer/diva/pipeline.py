"""DIVA framework card, paper tables, and smoke demos (arXiv:2605.25328)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.diva.config import DIVAConfig
from ltx_trainer.diva.factorization import DualFactorization, inject_logits, orthogonality_loss, pool_image_tokens
from ltx_trainer.diva.geometry import effective_rank_increment, reconstruction_residual
from ltx_trainer.diva.layout import LIMITATIONS
from ltx_trainer.diva.mock import make_toy_flows, random_mask_ratio
from ltx_trainer.diva.mutual_info import (
    directed_shared_alignment,
    nce_club_unique_upper_bound,
    stage2_total_loss,
)


def framework_card(cfg: DIVAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DIVAConfig()
    return {
        "name": "DIVA (Representation Divergence → Mutual Reinforcement)",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": (
            "Factorize middle-layer visual representations into shared (cross-task transfer) and unique "
            "(task-specific inductive bias) components from paired understanding + generation flows on the same anchor."
        ),
        "flows": {
            "understanding": "Image + captioning prompt → low-frequency semantic bias",
            "generation": "Masked inpainting (random mask r∈[0.2,0.6]) with paired text → high-frequency detail bias",
        },
        "stages": {
            "stage1": "Freeze backbone; train gated shared/unique encoders + low-rank logit readouts (Eq. 8–9)",
            "stage2": "Freeze encoders; fine-tune backbone with asymmetric MI alignment + unique regularization (Eq. 10–11)",
        },
        "mid_layers": f"{cfg.mid_layer_start}–{cfg.mid_layer_end}",
        "lambda_uni": cfg.lambda_uni,
        "umm_backbones": list(cfg.umm_backbones),
        "reported_gains": {"understanding_pct": 7.82, "generation_pct": 8.46},
    }


def table_main_results() -> dict[str, dict[str, float | str]]:
    """Table 1 excerpt — +DIVA deltas on single-backbone UMMs."""
    return {
        "Nexus-Gen": {
            "params": "7B AR",
            "MMMU": 43.5,
            "MMMU+DIVA": 49.4,
            "MME": 1279.1,
            "MME+DIVA": 1355.3,
            "MMBench": 70.7,
            "MMBench+DIVA": 74.9,
            "POPE": 83.6,
            "POPE+DIVA": 87.4,
            "GenEval": 0.77,
            "GenEval+DIVA": 0.83,
            "DPG-Bench": 81.30,
            "DPG-Bench+DIVA": 87.87,
            "WISE": 0.39,
            "WISE+DIVA": 0.45,
        },
        "Show-o": {
            "params": "1.5B AR",
            "MMMU": 26.3,
            "MMMU+DIVA": 32.4,
            "MME": 1097.7,
            "MME+DIVA": 1206.1,
            "MMBench": 48.7,
            "MMBench+DIVA": 51.0,
            "POPE": 73.1,
            "POPE+DIVA": 79.1,
            "GenEval": 0.57,
            "GenEval+DIVA": 0.64,
            "DPG-Bench": 69.81,
            "DPG-Bench+DIVA": 76.03,
            "WISE": 0.29,
            "WISE+DIVA": 0.34,
        },
        "Liquid": {
            "params": "7B AR",
            "MMMU": 30.2,
            "MMMU+DIVA": 34.0,
            "MME": 1321.7,
            "MME+DIVA": 1434.9,
            "MMBench": 57.2,
            "MMBench+DIVA": 58.9,
            "POPE": 77.4,
            "POPE+DIVA": 84.5,
            "GenEval": 0.70,
            "GenEval+DIVA": 0.81,
            "DPG-Bench": 80.63,
            "DPG-Bench+DIVA": 83.47,
            "WISE": 0.41,
            "WISE+DIVA": 0.44,
        },
    }


def table_ablation_show_o() -> dict[str, dict[str, float]]:
    """Table 2 — Show-o ablations."""
    return {
        "Base": {"MMMU": 26.3, "POPE": 73.1, "GenEval": 0.69, "DPG-Bench": 69.81},
        "Base+SFT": {"MMMU": 26.8, "POPE": 74.5, "GenEval": 0.67, "DPG-Bench": 70.75},
        "Base+DIVA": {"MMMU": 32.4, "POPE": 79.1, "GenEval": 0.75, "DPG-Bench": 76.03},
        "w/o Iuni": {"MMMU": 28.3, "POPE": 75.8, "GenEval": 0.70, "DPG-Bench": 71.58},
        "w/o sg[·]": {"MMMU": 31.7, "POPE": 78.2, "GenEval": 0.73, "DPG-Bench": 74.92},
        "Mid-Layer (8–18)": {"MMMU": 32.4, "POPE": 79.1, "GenEval": 0.75, "DPG-Bench": 76.03},
        "Linear+LN encoders": {"MMMU": 29.4, "POPE": 75.9, "GenEval": 0.71, "DPG-Bench": 72.37},
        "Contiguous mask": {"MMMU": 24.7, "POPE": 69.6, "GenEval": 0.70, "DPG-Bench": 68.22},
    }


def table_post_training_comparison() -> dict[str, dict[str, float]]:
    """Table 6 — vs RecA on Show-o."""
    return {
        "RecA": {"MME_delta": 37.1, "POPE_delta": 2.6, "GenEval_delta": 0.06},
        "DIVA": {"MME_delta": 108.4, "POPE_delta": 6.0, "GenEval_delta": 0.07},
    }


def table_lambda_uni_sensitivity() -> dict[str, dict[str, float]]:
    """Table 4 — λ_uni sensitivity on Show-o."""
    return {
        "Base": {"POPE": 73.1, "MMMU": 26.3, "DPG-Bench": 69.81, "GenEval": 0.69},
        "Base+SFT": {"POPE": 74.5, "MMMU": 26.8, "DPG-Bench": 70.75, "GenEval": 0.67},
        "lambda_uni=0.4": {"POPE": 78.3, "MMMU": 31.9, "DPG-Bench": 74.92, "GenEval": 0.74},
        "lambda_uni=0.6": {"POPE": 79.1, "MMMU": 32.4, "DPG-Bench": 76.03, "GenEval": 0.75},
        "lambda_uni=0.8": {"POPE": 78.7, "MMMU": 32.2, "DPG-Bench": 75.50, "GenEval": 0.75},
    }


def stage1_demo(cfg: DIVAConfig | None = None) -> dict[str, Any]:
    """Stage 1 smoke: factorization + orthogonality + logit injection."""
    cfg = cfg or DIVAConfig()
    samples = make_toy_flows(batch=3, hidden_dim=cfg.hidden_dim, seed=42)
    enc_u = DualFactorization(cfg.hidden_dim, cfg.factor_dim)
    enc_g = DualFactorization(cfg.hidden_dim, cfg.factor_dim)

    orth_vals: list[float] = []
    for s in samples:
        z_u_sh, z_u_uni = enc_u(s.h_u)
        z_g_sh, z_g_uni = enc_g(s.h_g)
        orth_vals.append(float(orthogonality_loss(z_u_sh, z_u_uni).item()))
        orth_vals.append(float(orthogonality_loss(z_g_sh, z_g_uni).item()))

    # Low-rank readouts as random projections for smoke
    r = cfg.readout_rank
    vt, vv = 8, 6
    a_u = torch.randn(vt, r) * 0.01
    a_g = torch.randn(vv, r) * 0.01
    b_u = torch.randn(vt, r) * 0.01
    b_g = torch.randn(vv, r) * 0.01
    s_u = torch.randn(vt)
    s_g = torch.randn(vv)
    z_u_sh, z_u_uni = enc_u(samples[0].h_u)
    z_g_sh, z_g_uni = enc_g(samples[0].h_g)
    tilde_u, tilde_g = inject_logits(s_u, s_g, z_g_sh[:r], z_u_uni[:r], z_u_sh[:r], z_g_uni[:r], a_u, a_g, b_u, b_g)

    return {
        "n_samples": len(samples),
        "orthogonality_mean": sum(orth_vals) / len(orth_vals),
        "logit_injection": {"tilde_u_shape": list(tilde_u.shape), "tilde_g_shape": list(tilde_g.shape)},
        "mask_ratio_sample": random_mask_ratio(low=cfg.mask_ratio_min, high=cfg.mask_ratio_max, generator=torch.Generator().manual_seed(0)),
    }


def stage2_demo(cfg: DIVAConfig | None = None) -> dict[str, Any]:
    """Stage 2 smoke: asymmetric shared alignment + unique bound + total loss."""
    cfg = cfg or DIVAConfig()
    samples = make_toy_flows(batch=4, hidden_dim=cfg.hidden_dim, seed=7)
    enc_u = DualFactorization(cfg.hidden_dim, cfg.factor_dim)
    enc_g = DualFactorization(cfg.hidden_dim, cfg.factor_dim)

    z_u_sh_list: list[torch.Tensor] = []
    z_g_sh_list: list[torch.Tensor] = []
    z_u_uni_list: list[torch.Tensor] = []
    z_g_uni_list: list[torch.Tensor] = []
    for s in samples:
        zu_sh, zu_uni = enc_u(s.h_u)
        zg_sh, zg_uni = enc_g(s.h_g)
        z_u_sh_list.append(zu_sh)
        z_g_sh_list.append(zg_sh)
        z_u_uni_list.append(zu_uni)
        z_g_uni_list.append(zg_uni)

    neg_g = [z_g_sh_list[1], z_g_sh_list[2], z_g_sh_list[3]]
    neg_u = [z_u_sh_list[1], z_u_sh_list[2], z_u_sh_list[3]]
    l_u2g = directed_shared_alignment(z_u_sh_list[0], z_g_sh_list[0], neg_g, temperature=cfg.temperature)
    l_g2u = directed_shared_alignment(z_g_sh_list[0], z_u_sh_list[0], neg_u, temperature=cfg.temperature)
    l_uni = nce_club_unique_upper_bound(z_u_uni_list[0], z_g_uni_list[0], z_g_uni_list[1])
    l_und = torch.tensor(0.5)
    l_gen = torch.tensor(0.6)
    total = stage2_total_loss(l_u2g, l_g2u, l_uni, l_und, l_gen, lambda_uni=cfg.lambda_uni)

    # Geometry on stacked flows
    h_u = torch.stack([s.h_u for s in samples])
    h_g = torch.stack([s.h_g for s in samples])
    h_cat = torch.cat([h_u, h_g], dim=0)
    r_res = reconstruction_residual(h_g, h_u)
    delta_er = effective_rank_increment(h_u, h_g, h_cat)

    return {
        "losses": {
            "L_U2G": float(l_u2g.item()),
            "L_G2U": float(l_g2u.item()),
            "L_uni": float(l_uni.item()),
            "L_total": float(total.item()),
        },
        "geometry": {"R_res_G_given_U": r_res, "delta_ER": delta_er},
    }


def evaluation_demo(cfg: DIVAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DIVAConfig()
    return {
        "framework": framework_card(cfg),
        "stage1": stage1_demo(cfg),
        "stage2": stage2_demo(cfg),
        "limitations": list(LIMITATIONS),
        "paper_tables": {
            "main_results": table_main_results(),
            "ablation_show_o": table_ablation_show_o(),
            "post_training_comparison": table_post_training_comparison(),
            "lambda_uni": table_lambda_uni_sensitivity(),
        },
    }


def training_step_demo(cfg: DIVAConfig | None = None) -> dict[str, Any]:
    """Single reference training step combining stage-1 orthogonality + stage-2 MI terms."""
    cfg = cfg or DIVAConfig()
    s1 = stage1_demo(cfg)
    s2 = stage2_demo(cfg)
    return {"stage1": s1, "stage2": s2, "mid_layers": [cfg.mid_layer_start, cfg.mid_layer_end]}
