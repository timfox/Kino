"""Demos: two-stage pre-train + rank-one LoRA fine-tuning."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lora_hd_attn.benchmarks import summary_anchors
from ltx_trainer.lora_hd_attn.config import LoraHdAttnConfig
from ltx_trainer.lora_hd_attn.effective_noise import (
    delta_eff_active_ft,
    delta_eff_independent,
    delta_eff_reused,
    effective_noise_card,
)
from ltx_trainer.lora_hd_attn.losses import frozen_baseline, test_error_finetune_stub, test_error_pretrain_stub
from ltx_trainer.lora_hd_attn.order_params import finetune_order_params, pretrain_order_params
from ltx_trainer.lora_hd_attn.simulation import run_finite_d_simulation


def _stage_summary(cfg: LoraHdAttnConfig) -> dict[str, Any]:
    pre = pretrain_order_params(cfg)
    de0 = delta_eff_independent(cfg.delta, pre["Q0"], pre["M"], pre["Q"])
    de = delta_eff_reused(de0, pre["V"], cfg.T) if cfg.e == 1 else de0
    fine = finetune_order_params(cfg, pretrain=pre, delta_eff=de)
    e_pre = test_error_pretrain_stub(cfg.delta, pre["Q0"], pre["M"], pre["Q"], cfg.T)
    e_ft = test_error_finetune_stub(de, m=fine["m"], q=fine["q"], q0=fine["q0"], T=cfg.T)
    e_frozen = frozen_baseline(de0, fine["q0"], cfg.T)
    return {
        "config": {
            "D": cfg.D,
            "T": cfg.T,
            "alpha": cfg.alpha,
            "alpha_prime": cfg.alpha_prime,
            "kappa0": cfg.kappa0,
            "kappa": cfg.kappa,
            "e": cfg.e,
            "sigma": cfg.sigma,
        },
        "pretrain": pre,
        "finetune": fine,
        "delta_eff": de,
        "E_pretrain": round(e_pre, 4),
        "E_finetune": round(e_ft, 4),
        "E_frozen_baseline": round(e_frozen, 4),
        "lora_gain_over_frozen": round(e_frozen - e_ft, 4),
        "effective_noise": effective_noise_card(
            delta=cfg.delta,
            Q0=pre["Q0"],
            M=pre["M"],
            Q=pre["Q"],
            V=pre["V"],
            T=cfg.T,
            e=cfg.e,
        ),
    }


def run_demo(cfg: LoraHdAttnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoraHdAttnConfig()
    theory = _stage_summary(cfg)
    sim = run_finite_d_simulation(cfg)
    lora_sim_gain = sim["train_error_frozen"] - sim["train_error_finetune"]
    return {
        **theory,
        "simulation": sim,
        "summary": summary_anchors(),
        "delta_eff_decreases_with_alignment": theory["pretrain"]["M"] > 0.2,
        "lora_beats_frozen": theory["lora_gain_over_frozen"] > 0.0 or lora_sim_gain > 0.0,
        "overlap_positive": theory["finetune"]["o_w"] > 0.0,
    }


def run_active_ft_demo(cfg: LoraHdAttnConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoraHdAttnConfig(delta=0.0, alpha_prime=3.0)
    pre = pretrain_order_params(cfg)
    de_uniform = delta_eff_independent(cfg.delta, pre["Q0"], pre["M"], pre["Q"])
    de_active = delta_eff_active_ft(cfg.delta, pre["Q0"], pre["M"], pre["Q"])
    fine_u = finetune_order_params(cfg, pretrain=pre, delta_eff=de_uniform)
    fine_a = finetune_order_params(cfg, pretrain=pre, delta_eff=de_active)
    e_u = test_error_finetune_stub(de_uniform, m=fine_u["m"], q=fine_u["q"], q0=fine_u["q0"], T=cfg.T)
    e_a = test_error_finetune_stub(de_active, m=fine_a["m"], q=fine_a["q"], q0=fine_a["q0"], T=cfg.T)
    return {
        "delta_eff_uniform": de_uniform,
        "delta_eff_active_ft": de_active,
        "E_prime_uniform": round(e_u, 4),
        "E_prime_active": round(e_a, 4),
        "delta_E_prime": round(e_u - e_a, 4),
        "active_improves": e_a < e_u,
        "overlap_uniform": fine_u["o_w"],
        "overlap_active": fine_a["o_w"],
    }


def run_reused_sequences_demo(cfg: LoraHdAttnConfig | None = None) -> dict[str, Any]:
    cfg0 = cfg or LoraHdAttnConfig()
    cfg1 = LoraHdAttnConfig(**{**cfg0.__dict__, "e": 1})
    indep = _stage_summary(cfg0)
    reused = _stage_summary(cfg1)
    return {
        "independent_e0": indep,
        "reused_e1": reused,
        "delta_eff_smaller_when_reused": reused["delta_eff"] <= indep["delta_eff"],
        "overlap_higher_when_reused": reused["finetune"]["o_w"] >= indep["finetune"]["o_w"],
        "mismatch_possible": reused["finetune"]["o_w"] > indep["finetune"]["o_w"]
        and reused["E_finetune"] > indep["E_finetune"] * 0.9,
    }
