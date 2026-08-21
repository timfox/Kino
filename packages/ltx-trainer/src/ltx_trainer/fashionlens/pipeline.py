"""FashionLens framework card, U-FIRE / Table II–III stubs, and smoke demos (arXiv:2605.22552)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fashionlens.config import FashionLensConfig
from ltx_trainer.fashionlens.ggas import (
    argmax_task_index,
    ema_difficulty,
    retrieval_token_difficulty,
    sampling_probabilities,
    sampling_score,
)
from ltx_trainer.fashionlens.layout import LIMITATIONS
from ltx_trainer.fashionlens.losses import infonce_retrieval_loss, mean_reciprocal_rank_from_r_at, total_training_loss
from ltx_trainer.fashionlens.mock import toy_unit_query, toy_unit_target
from ltx_trainer.fashionlens.pgsqc import (
    adaptation_proposal,
    frobenius_squared,
    interpolation_lambda,
    orthogonality_loss_frobenius,
    slerp,
)
from ltx_trainer.fashionlens.ufire import ufire_scale_summary, ufire_task_rubric


def framework_card(cfg: FashionLensConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FashionLensConfig()
    return {
        "name": "FashionLens",
        "paper": cfg.paper_arxiv,
        "code": cfg.code_url,
        "benchmark": "U-FIRE (Unified Fashion Image Retrieval & Evaluation)",
        "idea": (
            "MLLM backbone with learnable [RETq]/[RETt] tokens; PGSQC adapts queries on the "
            "hypersphere via low-rank proposal + adaptive Slerp; GGAS re-weights tasks using "
            "retrieval-token gradient norms with size-aware softmax."
        ),
        "backbone": cfg.backbone,
        "pgsqc": {"embed_dim_D": cfg.embed_dim_D, "low_rank_d": cfg.low_rank_d},
        "ggas": {
            "ema_alpha": cfg.ema_alpha,
            "gamma": cfg.ggas_gamma,
            "eta": cfg.ggas_eta,
            "min_sample_eps": cfg.min_sample_eps,
        },
        "training": {
            "lora_rank": cfg.lora_rank,
            "epochs": cfg.train_epochs,
            "lr": cfg.lr,
            "gradcache_batch": cfg.gradcache_effective_batch,
            "infonce_tau": cfg.infonce_tau,
        },
    }


def table_ii_main_results() -> list[dict[str, str | float]]:
    """Table II — mR = (R@1+R@5+R@10)/3; FashionLens column (excerpt)."""
    return [
        {"task": 1, "dataset": "FashionGen", "fashionlens": 59.45, "gme": 37.80, "qwen3_vl": 46.41},
        {"task": 1, "dataset": "Shoes", "fashionlens": 52.06, "gme": 45.61, "qwen3_vl": 44.17},
        {"task": 2, "dataset": "HAIFashion", "fashionlens": 92.17, "gme": 53.33, "qwen3_vl": 54.84},
        {"task": 3, "dataset": "DeepFashion2", "fashionlens": 79.26, "gme": 32.19, "qwen3_vl": 72.65},
        {"task": 5, "dataset": "MovingFashion", "fashionlens": 84.16, "gme": 44.51, "qwen3_vl": 78.44},
        {"task": 7, "dataset": "FashionIQ-Dress", "fashionlens": 27.37, "gme": 27.22, "qwen3_vl": 20.97},
        {"task": 9, "dataset": "FashionAI", "fashionlens": 76.31, "gme": 53.87, "qwen3_vl": 70.58},
        {"task": 10, "dataset": "DeepFashion2 (OOD)", "fashionlens": 66.27, "gme": 26.03, "qwen3_vl": 41.80},
        {"task": 11, "dataset": "Polyvore (OOD)", "fashionlens": 77.70, "gme": 63.13, "qwen3_vl": 57.83},
        {"task": "avg", "dataset": "Average (Commonly Supported)", "fashionlens": 51.24, "gme": 35.11, "qwen3_vl": 40.05},
        {"task": "avg", "dataset": "Average (All)", "fashionlens": 52.21, "gme": 35.40, "qwen3_vl": 41.35},
    ]


def table_iii_ablation() -> list[dict[str, str | float]]:
    """Table III — average over training tasks (R@1, R@5, R@10, mR)."""
    return [
        {"id": 1, "query_rep": "q0", "sampling": "Random", "r1": 22.62, "r5": 45.00, "r10": 53.61, "mR": 40.41},
        {"id": 2, "query_rep": "q0", "sampling": "GGAS", "r1": 30.76, "r5": 54.09, "r10": 62.08, "mR": 48.98},
        {"id": 3, "query_rep": "qp", "sampling": "GGAS", "r1": 31.08, "r5": 53.98, "r10": 61.90, "mR": 48.99},
        {"id": 4, "query_rep": "q (Linear)", "sampling": "GGAS", "r1": 31.00, "r5": 53.60, "r10": 61.27, "mR": 48.62},
        {"id": 5, "query_rep": "q (Slerp)", "sampling": "GGAS (w/o size)", "r1": 31.03, "r5": 54.31, "r10": 62.04, "mR": 49.12},
        {"id": 6, "query_rep": "q (shared params.)", "sampling": "GGAS", "r1": 31.41, "r5": 54.91, "r10": 62.36, "mR": 49.56},
        {
            "id": "ours",
            "query_rep": "q (Slerp)",
            "sampling": "GGAS",
            "r1": 32.17,
            "r5": 55.21,
            "r10": 62.65,
            "mR": 50.01,
        },
    ]


def pipeline_demo(cfg: FashionLensConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FashionLensConfig()
    q0 = toy_unit_query(dim=8)
    qp = adaptation_proposal(q0)
    lam = interpolation_lambda(q0)
    q_adapt = slerp(q0, qp, lam)
    # Toy low-rank A as 3×3 rows for orthogonality stub
    a_stub = [[0.2, 0.0, 0.1], [0.0, 0.25, 0.0], [0.1, 0.0, 0.2]]
    l_ortho = orthogonality_loss_frobenius(a_stub)
    l_reg = frobenius_squared(a_stub) * 1e-2
    t0 = toy_unit_target(dim=8)
    base_sim = sum(q_adapt[i] * t0[i] for i in range(len(q_adapt)))
    dots = [base_sim, 0.3 * base_sim, 0.1 * base_sim]
    l_ret = infonce_retrieval_loss(dots, temperature=cfg.infonce_tau, positive_index=0)
    total = total_training_loss(l_ret, l_ortho, l_reg, beta1=cfg.beta_ortho, beta2=cfg.beta_reg)
    d1 = retrieval_token_difficulty(1.2, 0.8)
    d2 = retrieval_token_difficulty(0.4, 0.3)
    g1 = ema_difficulty(1.0, d1, alpha=cfg.ema_alpha)
    g2 = ema_difficulty(0.5, d2, alpha=cfg.ema_alpha)
    s1 = sampling_score(g1, 49_084.0, eta=cfg.ggas_eta, gamma=cfg.ggas_gamma)
    s2 = sampling_score(g2, 1_600.0, eta=cfg.ggas_eta, gamma=cfg.ggas_gamma)
    probs = sampling_probabilities([s1, s2], eps=cfg.min_sample_eps)
    pick = argmax_task_index(probs)
    return {
        "slerp_lambda": lam,
        "adapted_query_norm": sum(x * x for x in q_adapt) ** 0.5,
        "infonce_scalar": l_ret,
        "ortho_loss_scalar": l_ortho,
        "total_loss_scalar": total,
        "ggas_sample_probs": probs,
        "ggas_argmax_index": pick,
    }


def evaluation_demo(cfg: FashionLensConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FashionLensConfig()
    ours = next(r for r in table_iii_ablation() if r["id"] == "ours")
    base = next(r for r in table_iii_ablation() if r["id"] == 1)
    delta_mr = float(ours["mR"]) - float(base["mR"])
    ood10 = next(r for r in table_ii_main_results() if r["dataset"] == "DeepFashion2 (OOD)")
    return {
        "framework": framework_card(cfg),
        "ufire": ufire_scale_summary(),
        "tasks": ufire_task_rubric(),
        "pipeline": pipeline_demo(cfg),
        "limitations": list(LIMITATIONS),
        "ablation_mR_gain_vs_base_random": round(delta_mr, 2),
        "ood_task10_mR": ood10["fashionlens"],
        "paper_tables": {
            "table_ii_excerpt": table_ii_main_results(),
            "table_iii_ablation": table_iii_ablation(),
        },
    }
