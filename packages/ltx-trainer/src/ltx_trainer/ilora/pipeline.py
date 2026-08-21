"""Framework card, demos, and benchmark manifest (arXiv:2605.30179)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.ilora.config import ILORAConfig
from ltx_trainer.ilora.graph import matched_poisson_rate, poisson_kl
from ltx_trainer.ilora.loss import ilora_total_loss, prediction_loss
from ltx_trainer.ilora.ltx_bridge import ILORALTXAdapterBridge, ltx_peft_integration_notes
from ltx_trainer.ilora.metrics import (
    table1_molweni,
    table2_graph_error,
    table3_ibd_diagnosis,
    table4_ablation,
    table5_tabular_baselines,
    table8_inference_cost,
)


def framework_card(cfg: ILORAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ILORAConfig()
    return {
        "name": "iLoRA",
        "paper": cfg.paper_arxiv,
        "task": "Bayesian graph-conditioned LoRA for microbiome IBD diagnosis and Molweni structure recovery",
        "backbones": {"ibd": cfg.ibd_backbone, "molweni": cfg.molweni_backbone},
        "components": [
            "Taxa selection S(X) → K entities",
            "Poisson interaction graph + Gaussian proxy (Theorem 5.1)",
            "NPN Poisson→Laplace sparsification",
            "GCN embedding → hypernetwork LoRA A matrix",
            "Monte Carlo graph marginalization (Eq. 13)",
        ],
        "ltx_bridge": "ILORALTXAdapterBridge: sample-conditioned LoRA A for DiT PEFT layers",
        "datasets": ["Molweni (discourse graphs)", "IBD UC vs CD (1014 samples, 3061 taxa)"],
        "selected_taxa_k": cfg.num_taxa,
    }


def paper_limitations() -> list[str]:
    return [
        "Graph branch scales O(K²); feature selection (MaAsLin2) may drop informative taxa.",
        "Inferred graphs are predictive/statistical, not causal microbial mechanisms.",
        "Full Qwen3/Llama fine-tuning and cohort downloads are external to this stub.",
        "LTX bridge is architectural guidance; not yet wired into LtxvTrainer._setup_lora.",
    ]


def benchmark_manifest(cfg: ILORAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ILORAConfig()
    return {
        "molweni": {"metrics": ["F1", "EM"], "graph_error_vs_random": "26.7% vs 50.0%"},
        "ibd": {
            "metrics": ["ECE", "F1_UC", "AUROC", "AUPRC"],
            "split": "710/152/152 stratified by cohort",
            "features": f"top-{cfg.num_taxa} MaAsLin2 species",
        },
        "baselines": ["MLE", "MAP", "MCD", "ENS", "BLOB", "LAP"],
        "inference_cost": table8_inference_cost(),
    }


def evaluation_demo(*, cfg: ILORAConfig | None = None, k: int = 8, d_in: int = 64) -> dict[str, Any]:
    cfg = cfg or ILORAConfig()
    torch.manual_seed(7)
    bridge = ILORALTXAdapterBridge(cfg, d_in=d_in, d_out=d_in)
    abundances = torch.softmax(torch.randn(2, k), dim=-1)
    out = bridge(abundances)
    logits = torch.randn(2, 2)
    labels = torch.tensor([0, 1])
    pred = prediction_loss(logits, labels)
    total = ilora_total_loss(
        pred,
        out["pois_kl"],
        out["lap_kl"],
        lambda_pois=cfg.lambda_pois,
        lambda_lap=cfg.lambda_lap,
    )
    u = torch.tensor([1.2, 0.3])
    delta = torch.tensor([0.4, 0.2])
    m = matched_poisson_rate(u, delta)
    kl_demo = float(poisson_kl(m, m * 0.9).mean().item())
    ilora_row = next(r for r in table3_ibd_diagnosis() if r["method"] == "iLoRA")
    mle_row = next(r for r in table3_ibd_diagnosis() if r["method"] == "MLE")
    return {
        "lora_a_shape": list(out["lora_a"].shape),
        "adjacency_density": round(float((out["adjacency"] > 0).float().mean().item()), 4),
        "total_loss": round(float(total.item()), 4),
        "matched_poisson_rate": round(float(m.mean().item()), 4),
        "poisson_kl_demo": round(kl_demo, 4),
        "ibd_auroc_gain_vs_mle": round(float(ilora_row["AUROC"]) - float(mle_row["AUROC"]), 4),
        "ltx_integration": ltx_peft_integration_notes(cfg),
    }


def training_step_demo(*, cfg: ILORAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ILORAConfig()
    bridge = ILORALTXAdapterBridge(cfg, d_in=32, d_out=32)
    opt = torch.optim.AdamW(bridge.parameters(), lr=2e-4)
    x = torch.softmax(torch.randn(1, cfg.num_taxa), dim=-1)
    out = bridge(x)
    loss = out["pois_kl"] + out["lap_kl"] + out["lora_a"].pow(2).mean() * 1e-3
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {
        "step_loss": round(float(loss.item()), 4),
        "trainable_params": sum(p.numel() for p in bridge.parameters() if p.requires_grad),
    }
