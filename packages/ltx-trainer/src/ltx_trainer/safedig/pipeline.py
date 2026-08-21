"""Framework card, demos, and benchmark manifest (arXiv:2605.30049)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.safedig.config import SafeDIGConfig
from ltx_trainer.safedig.metrics import (
    fuse_unsafe_flags,
    line_level_asr,
    prompt_level_asr,
    table1_line_level,
    table1_main_results,
    table2_transfer_positions,
    table6_ablation_snippet,
)
from ltx_trainer.safedig.routing import rank_interventions
from ltx_trainer.safedig.sae import (
    contrast_activation,
    decoder_only_transfer,
    default_hooks,
    train_source_sae,
    SparseAutoencoder,
)
from ltx_trainer.safedig.steering import estimate_harmful_mask, steer_activation


def framework_card(cfg: SafeDIGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SafeDIGConfig()
    return {
        "name": "SafeDIG",
        "paper": cfg.paper_arxiv,
        "task": "Robust generalizable safety steering for text-to-image DiTs",
        "backbones": list(cfg.backbones),
        "components": [
            "Position-aware SAE hooks (Ltext, Lbind, Lrender)",
            "Robustness-aware routing R(ℓ, o)",
            "Manifold-stable decoder-only transfer",
            "Inference Blend / Repel steering",
        ],
        "roles": ["SAE encoder (frozen dictionary)", "SAE decoder (target adapter)", "Router"],
        "benchmark": cfg.benchmark_name,
        "transfer_protocol": {
            "source": list(cfg.source_categories),
            "target": cfg.target_category,
        },
        "evaluators": ["Q16", "NudeNet"],
        "asr_metrics": ["Prompt-level ASR", "Line-level ASR"],
    }


def paper_limitations() -> list[str]:
    return [
        "Intervention effect not localized to attention vs MLP vs timestep in this stub.",
        "Safety-contrast a+ − a− depends on paired prompt rewriting quality.",
        "Full FLUX/SD3.5 activation caching and Cycles-scale training are external.",
    ]


def benchmark_manifest(cfg: SafeDIGConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SafeDIGConfig()
    return {
        "primary": "i2p (4703 prompts, 7 categories)",
        "transfer": f"source: 6 categories excluding {cfg.target_category}; target: {cfg.target_category}",
        "stress_tests": ["MMA adversarial jailbreak", "MM-SafetyBench (appendix)"],
        "samples_per_prompt": cfg.images_per_prompt,
        "baselines": ["SAFREE", "EraseDiff", "Erasing", "SAeUron"],
    }


def evaluation_demo(*, cfg: SafeDIGConfig | None = None, dim: int = 128) -> dict[str, Any]:
    cfg = cfg or SafeDIGConfig()
    torch.manual_seed(42)
    hooks = default_hooks(cfg)
    sae = SparseAutoencoder(dim, expansion=4, sparsity_lambda=cfg.sparsity_lambda)

    a_safe = torch.randn(32, dim) * 0.5
    a_harm = torch.randn(32, dim) * 0.5 + 0.3
    delta = contrast_activation(a_safe, a_harm)
    src_loss = train_source_sae(sae, delta, steps=8)
    tgt = torch.randn(16, dim) * 0.4 + 0.1
    xfer_loss = decoder_only_transfer(sae, tgt, replay_activations=delta, steps=4)

    cat_emb = torch.randn(dim)
    lat_emb = torch.randn(dim)
    delta_map = {h.name: delta[: min(32, delta.shape[0])] for h in hooks}
    ranked = rank_interventions(
        hooks,
        cfg.operators,
        delta_a_by_hook=delta_map,
        category_emb=cat_emb,
        latent_emb=lat_emb,
        cfg=cfg,
    )
    top = ranked[0]
    with torch.no_grad():
        z, a_hat = sae(a_harm[:1])
        a_steered = steer_activation(
            a_harm[:1],
            sae,
            a_hat,
            operator=top.operator,
            strength=cfg.default_blend_beta,
            harm_mask=estimate_harmful_mask(sae, a_harm),
        )

    flags_qn = [[(False, True), (False, False)], [(True, False), (False, False)], [(False, False), (True, False)]]
    fused = [[fuse_unsafe_flags(q, n) for q, n in row] for row in flags_qn]

    return {
        "source_sae_loss": round(src_loss, 4),
        "transfer_loss": round(xfer_loss, 4),
        "top_hook": top.hook.name,
        "top_operator": top.operator,
        "routing_score": round(top.score, 4),
        "steered_delta_norm": round(float((a_steered - a_harm[:1]).norm().item()), 4),
        "line_asr_demo": round(line_level_asr(f for row in fused for f in row), 3),
        "prompt_asr_demo": round(prompt_level_asr(fused), 3),
        "table1_best_delta_flux": 15.96,
    }


def training_step_demo(*, cfg: SafeDIGConfig | None = None) -> dict[str, Any]:
    """End-to-end smoke: route → train → transfer → steer."""
    return evaluation_demo(cfg=cfg)
