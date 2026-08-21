"""Parametric Memory Law framework card and demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.parametric_memory.config import ParametricMemoryConfig
from ltx_trainer.parametric_memory.layout import LIMITATIONS
from ltx_trainer.parametric_memory.lora import LoRALinear, effective_lora_params
from ltx_trainer.parametric_memory.memft import first_failure_index, memft_ot_loss, memft_sw_seq_weights, sequence_loss
from ltx_trainer.parametric_memory.memory_law import (
    delta_loss,
    fit_law_grid,
    loss_from_probability,
    parametric_memory_law,
    probability_from_loss,
)
from ltx_trainer.parametric_memory.curriculum import exposure_at_epoch, phonebook_curriculum_qwen
from ltx_trainer.parametric_memory.knowledge import knowledge_card, references_bibtex
from ltx_trainer.parametric_memory.paper_tables import (
    table1_law_fit,
    table2_memft_longcontext_acctok,
    table2_phonebook_accem,
    table3_linear_rule_generalization,
    table4_exact_memory_scenarios,
)
from ltx_trainer.parametric_memory.ranks import longcontext_ranks
from ltx_trainer.parametric_memory.phase_transition import (
    L_CRIT,
    exact_match_accuracy,
    is_ordered_phase,
    token_accuracy,
)


def framework_card(cfg: ParametricMemoryConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ParametricMemoryConfig()
    return {
        "name": "Parametric Memory Law (How LoRA Remembers)",
        "paper": cfg.paper_arxiv,
        "authors": cfg.authors,
        "code_url": cfg.code_url,
        "idea": (
            "LoRA as a latent-space memory probe: ΔL scales as C·r^α·ℓ^(-β). "
            "Token-level phase transition at L_crit=ln(2) (p>0.5). MemFT redirects "
            "gradients to sub-threshold stubborn tokens."
        ),
        "law": "ΔL(r, ℓ) = C · r^α · ℓ^(-β) + b",
        "l_crit": cfg.l_crit,
        "p_threshold": cfg.p_threshold,
        "models": list(cfg.models),
        "memft_variants": list(cfg.memft_variants),
        "limitations": list(LIMITATIONS),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_law_fit": table1_law_fit(),
        "table2_longcontext_acctok": table2_memft_longcontext_acctok(),
        "table2_phonebook_accem": table2_phonebook_accem(),
        "table3_linear_rule": table3_linear_rule_generalization(),
        "table4_exact_memory": table4_exact_memory_scenarios(),
        "phonebook_curriculum_qwen": phonebook_curriculum_qwen(),
    }


def knowledge_bundle() -> dict[str, Any]:
    return {
        "knowledge": knowledge_card(),
        "references_bibtex": references_bibtex(),
        "qwen_longcontext_ranks": list(longcontext_ranks("Qwen3-8B-IT")),
    }


def evaluation_demo(cfg: ParametricMemoryConfig | None = None, *, seed: int = 7) -> dict[str, Any]:
    cfg = cfg or ParametricMemoryConfig()
    torch.manual_seed(seed)

    ranks = [1, 2, 4, 8]
    lengths = [100, 500, 2000]
    delta_grid = [[parametric_memory_law(r, ell, C=cfg.law_C, alpha=cfg.law_alpha, beta=cfg.law_beta, b=cfg.law_b).item()
                   for ell in lengths] for r in ranks]
    fit = fit_law_grid(ranks, lengths, delta_grid)

    # Phase transition toy
    p_easy = torch.tensor([0.9, 0.85, 0.95])
    p_hard = torch.tensor([0.3, 0.45, 0.48])
    l_easy = loss_from_probability(p_easy)
    l_hard = loss_from_probability(p_hard)
    ordered_frac = float(is_ordered_phase(torch.cat([l_easy, l_hard])).float().mean().item())

    # MemFT on synthetic token losses
    token_losses = torch.cat([l_easy, l_hard])
    l_sft = sequence_loss(token_losses)
    l_ot = memft_ot_loss(token_losses, cfg.l_crit)
    anchor = first_failure_index(
        torch.tensor([0, 1, 2, 0, 1, 2]),
        torch.tensor([0, 1, 2, 9, 9, 9]),
    )
    w_sw = memft_sw_seq_weights(token_losses, anchor)
    l_sw = (w_sw * token_losses).sum() / (w_sw.sum() + 1e-8)

    # LoRA smoke
    layer = LoRALinear(32, 64, rank=4)
    x = torch.randn(8, 32)
    _ = layer(x)
    eff = effective_lora_params(4, 32, 64)

    pred = torch.tensor([0, 1, 2, 3, 4])
    tgt = torch.tensor([0, 1, 2, 3, 5])
    acctok = token_accuracy(pred, tgt)
    accem = exact_match_accuracy(pred, tgt)

    d_l = delta_loss(torch.tensor(1.2), torch.tensor(0.4))
    t1 = table1_law_fit()
    qwen_ot = table2_memft_longcontext_acctok()["Qwen3-8B-IT"]["MemFT-OT"][-1]
    cur = phonebook_curriculum_qwen()[0]
    exposure_25 = exposure_at_epoch(25, cur["boundaries"])

    return {
        "law_predict_r8_l2k": parametric_memory_law(8, 2000, C=cfg.law_C, alpha=cfg.law_alpha, beta=cfg.law_beta, b=cfg.law_b).item(),
        "fit_r2": fit["r2"],
        "l_crit": L_CRIT,
        "ordered_fraction": ordered_frac,
        "loss_sft": float(l_sft.item()),
        "loss_memft_ot": float(l_ot.item()),
        "loss_memft_sw": float(l_sw.item()),
        "lora_effective_params_r4": eff,
        "delta_L_smoke": float(d_l.item()),
        "token_acc": acctok,
        "exact_match": accem,
        "first_failure_idx": anchor,
        "table1_qwen_comb_r2": t1[2]["r2"],
        "qwen_memft_ot_r9_acctok": qwen_ot,
        "p_at_lcrit": float(probability_from_loss(torch.tensor(cfg.l_crit)).item()),
        "phonebook_exposure_epoch25": exposure_25,
        "table4_domains": len(table4_exact_memory_scenarios()),
    }
