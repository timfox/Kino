"""R-FUML framework card, paper tables, and smoke demos (arXiv:2605.24475)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.rfuml.config import RFUMLConfig
from ltx_trainer.rfuml.conflict import conflict_for_view
from ltx_trainer.rfuml.fuzzy import (
    category_credibility,
    logits_to_memberships,
    uncertainty_from_credibility,
)
from ltx_trainer.rfuml.fusion import fuse_memberships, rmf_weights
from ltx_trainer.rfuml.layout import LIMITATIONS
from ltx_trainer.rfuml.loss import lccl, total_loss
from ltx_trainer.rfuml.mock import instance_memberships, make_clean_instance, make_conflicting_instance
from ltx_trainer.rfuml.rlvc import cyclical_learning_rate, fit_gmm_two_component


def framework_card(cfg: RFUMLConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RFUMLConfig()
    return {
        "name": "R-FUML (Robust Fuzzy Multi-View Learning)",
        "paper": cfg.paper_arxiv,
        "problem": "Trusted multi-view classification under view conflict (VC) in training and testing",
        "components": {
            "fuzzy": "Memberships → category credibility → entropy uncertainty (Eq. 3–4)",
            "RMF": "Down-weight high-uncertainty and high-conflict views (Eq. 6–7)",
            "RLVC": "Pre-train → cyclical training → GMM partition → robust re-train (Eq. 11–16)",
        },
        "datasets": list(cfg.datasets),
        "vc_protocol": "Mislabel V−2 random views at rate r; Gaussian noise on 10% test instances",
        "baselines_count": cfg.n_baselines,
        "ltx_note": "Use RMF weights when fusing per-modality losses/embeddings with suspected cross-view conflict.",
    }


def table_accuracy_excerpt() -> dict[str, dict[int, dict[str, float]]]:
    """Table I excerpt — R-FUML vs FUML (mean accuracy %) on four datasets."""
    return {
        "HW": {
            0: {"FUML": 99.20, "R-FUML": 99.12},
            40: {"FUML": 93.35, "R-FUML": 95.35},
            60: {"FUML": 89.12, "R-FUML": 92.32},
        },
        "Fashion": {
            0: {"FUML": 98.96, "R-FUML": 99.02},
            40: {"FUML": 94.82, "R-FUML": 95.00},
            60: {"FUML": 92.72, "R-FUML": 92.89},
        },
        "Scene": {
            0: {"FUML": 79.41, "R-FUML": 79.46},
            40: {"FUML": 73.27, "R-FUML": 74.19},
            60: {"FUML": 69.30, "R-FUML": 71.40},
        },
        "LandUse": {
            0: {"FUML": 76.71, "R-FUML": 77.00},
            40: {"FUML": 66.12, "R-FUML": 68.00},
            60: {"FUML": 61.00, "R-FUML": 64.33},
        },
    }


def table_fpr95() -> dict[str, dict[str, float]]:
    """Table II — FPR95 on clean vs conflicting samples (lower is better)."""
    return {
        "HW": {"TUNED": 0.852, "SAEML": 0.874, "FUML": 0.518, "R-FUML": 0.218},
        "Fashion": {"TUNED": 0.880, "SAEML": 0.923, "FUML": 0.410, "R-FUML": 0.086},
        "Scene": {"TUNED": 0.940, "SAEML": 0.934, "FUML": 0.604, "R-FUML": 0.490},
        "LandUse": {"TUNED": 0.937, "SAEML": 0.937, "FUML": 0.703, "R-FUML": 0.573},
    }


def table_ablation_loss_fusion() -> dict[str, dict[str, float]]:
    """Table III — 40% VC ablation (accuracy %); full R-FUML row (La+Lv+RMF)."""
    return {
        "HW": {"Concat_baseline": 86.22, "Avg_partial": 94.62, "DRF": 95.35, "RMF_full": 95.35},
        "Fashion": {"Concat_baseline": 90.06, "Avg_partial": 94.40, "DRF": 94.78, "RMF_full": 95.00},
        "Scene": {"Concat_baseline": 67.09, "Avg_partial": 74.02, "DRF": 73.98, "RMF_full": 74.19},
        "LandUse": {"Concat_baseline": 57.67, "Avg_partial": 67.83, "DRF": 67.71, "RMF_full": 68.00},
    }


def table_ablation_rlvc_stages() -> dict[str, dict[str, float]]:
    """Table IV — RLVC stage ablation at 40% VC (accuracy %)."""
    return {
        "HW": {"no_pre_no_cyc": 93.60, "pre_only": 94.68, "cyc_only": 95.00, "full_RLVC": 95.35},
        "Fashion": {"no_pre_no_cyc": 94.14, "pre_only": 94.22, "cyc_only": 94.84, "full_RLVC": 95.00},
        "Scene": {"no_pre_no_cyc": 73.86, "pre_only": 73.69, "cyc_only": 74.15, "full_RLVC": 74.19},
        "LandUse": {"no_pre_no_cyc": 66.40, "pre_only": 67.05, "cyc_only": 67.00, "full_RLVC": 68.00},
    }


def table_degradation_hw() -> dict[str, float]:
    """Sec. IV-B — accuracy drop from 0% to 60% VC on HW (paper narrative)."""
    return {
        "DCP-CV": 12.90,
        "PDF": 15.23,
        "TUNED": 15.98,
        "SAEML": 14.27,
        "FUML": 10.08,
        "R-FUML": 6.80,
    }


def rmf_demo() -> dict[str, Any]:
    """Smoke: clean vs conflicting toy instances — uncertainty, conflict, RMF weights."""
    clean = make_clean_instance()
    conflict = make_conflicting_instance()
    out: dict[str, Any] = {}
    for name, inst in ("clean", clean), ("conflicting", conflict):
        ms = instance_memberships(inst)
        us = [uncertainty_from_credibility(category_credibility(m)) for m in ms]
        os_ = [conflict_for_view(ms, v) for v in range(len(ms))]
        w_train = rmf_weights(us, os_, training=True)
        w_test = rmf_weights(us, os_, training=False)
        fused_test = fuse_memberships(ms, us, os_, training=False)
        out[name] = {
            "uncertainties": us,
            "conflicts": os_,
            "weights_train": w_train,
            "weights_test": w_test,
            "fused_membership": fused_test,
            "fused_uncertainty": uncertainty_from_credibility(category_credibility(fused_test)),
        }
    return out


def rlvc_demo() -> dict[str, Any]:
    """Smoke: cyclical LR + GMM on synthetic loss trajectories."""
    lr = [cyclical_learning_rate(e, eta_max=0.1, eta_min=0.001, cycle_length=10) for e in range(1, 21)]
    # Clean samples: low stable loss; conflicting: high volatile loss
    clean_losses = [0.15, 0.14, 0.16, 0.15, 0.14]
    conflict_losses = [0.85, 0.92, 0.78, 0.88, 0.95, 0.81, 0.90]
    all_losses = clean_losses + conflict_losses
    part = fit_gmm_two_component(all_losses)
    return {
        "lr_first_20_epochs": lr,
        "gmm_clean_count": len(part.clean_indices),
        "gmm_conflict_count": len(part.conflicting_indices),
        "posterior_clean": part.posterior_clean,
    }


def evaluation_demo(cfg: RFUMLConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RFUMLConfig()
    clean = make_clean_instance()
    ms = instance_memberships(clean)
    y = clean.true_label
    fused = fuse_memberships(
        ms,
        [uncertainty_from_credibility(category_credibility(m)) for m in ms],
        [conflict_for_view(ms, v) for v in range(len(ms))],
        training=False,
    )
    loss = total_loss(ms, fused, y, gamma=1.0)
    m_example = logits_to_memberships([0.8, 0.1, 0.1])
    c_example = category_credibility(m_example)
    return {
        "framework": framework_card(cfg),
        "membership_example": m_example,
        "credibility_example": c_example,
        "uncertainty_example": uncertainty_from_credibility(c_example),
        "rmf": rmf_demo(),
        "rlvc": rlvc_demo(),
        "total_loss_clean_smoke": loss,
        "paper_tables": {
            "accuracy_excerpt": table_accuracy_excerpt(),
            "fpr95": table_fpr95(),
            "ablation_fusion": table_ablation_loss_fusion(),
            "ablation_rlvc": table_ablation_rlvc_stages(),
            "degradation_hw": table_degradation_hw(),
        },
        "limitations": list(LIMITATIONS),
    }


def training_step_demo(cfg: RFUMLConfig | None = None) -> dict[str, Any]:
    """Single-step reference: fused loss + γ schedule for tooling parity."""
    cfg = cfg or RFUMLConfig()
    inst = make_conflicting_instance()
    ms = instance_memberships(inst)
    us = [uncertainty_from_credibility(category_credibility(m)) for m in ms]
    os_ = [conflict_for_view(ms, v) for v in range(len(ms))]
    fused = fuse_memberships(ms, us, os_, training=True)
    from ltx_trainer.rfuml.fuzzy import gamma_schedule

    return {
        "gamma_epoch_5": gamma_schedule(5, cfg.gamma_warmup_epochs),
        "lccl_per_view": [lccl(m, inst.true_label) for m in ms],
        "total_loss": total_loss(ms, fused, inst.true_label, gamma=0.5),
        "recommended_cycles": cfg.recommended_cycle_count,
    }
