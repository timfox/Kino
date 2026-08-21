"""LOPEO framework card and paper tables (IEEE SPL, Zhang et al.)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lopeo.balance import balance_index, counts_from_trials
from ltx_trainer.lopeo.config import LopeoConfig
from ltx_trainer.lopeo.cv import (
    pair_leakage_count,
    partition_trials_loeo,
    partition_trials_loto,
    partition_trials_lopeo,
)
from ltx_trainer.lopeo.datasets import (
    construct_dtu_trials,
    construct_kul_trials,
    construct_nju_ceegrid_trials,
    dataset_condition_summary,
)
from ltx_trainer.lopeo.layout import LIMITATIONS
from ltx_trainer.lopeo.mock import IdentityMemorizingDecoder, envelope_reconstruction_demo
from ltx_trainer.lopeo.stats import lopeo_mitigation_summary, significance_table
from ltx_trainer.lopeo.metrics import (
    contrastive_pcc_loss,
    decoding_accuracy,
    pcc_loss,
    pearson_corr,
    rho_delta,
)


def framework_card(cfg: LopeoConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LopeoConfig()
    return {
        "name": "LOPEO / EEG-AAD balance",
        "title": (
            "Decoding Stimulus Reconstruction-Based Auditory Attention "
            "Robustly in Unbalanced EEG Datasets"
        ),
        "venue": cfg.venue,
        "github": cfg.github_repo,
        "model": cfg.model,
        "idea": (
            "Stimulus reconstruction DNN decoders (VLAAI) overestimate decoding "
            "accuracy on unbalanced EEG-AAD datasets when the same envelope appears "
            "in both train and test under LOTO. LOPEO withholds entire attended–"
            "unattended stimulus pairs; LOEO is the NJU cEEGrid variant."
        ),
        "balance_index": "BI = (1/N_audio) Σ_j |n_att^j − n_unatt^j| / (n_att^j + n_unatt^j)",
        "losses": ["PCC (Eq. 4)", "contrastive PCC Δ (Eq. 5)"],
        "cv_strategies": ["LOTO", "LOPEO", "LOEO (3-speaker NJU)"],
        "datasets": ["KUL", "DTU", "NJU cEEGrid"],
        "training": {
            "optimizer": cfg.optimizer,
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
        },
        "defaults": cfg.__dict__,
    }


def table_i_datasets() -> list[dict[str, Any]]:
    """Table 1 — public EEG-AAD datasets used in the paper."""
    return [
        {
            "dataset": "KUL",
            "trials": 20,
            "subjects": 16,
            "speakers": 2,
            "balance_index": 0.600,
            "chance_acc": 0.5,
        },
        {
            "dataset": "DTU",
            "trials": 60,
            "subjects": 18,
            "speakers": 2,
            "balance_index": 0.056,
            "chance_acc": 0.5,
        },
        {
            "dataset": "NJU cEEGrid",
            "trials": 63,
            "subjects": 98,
            "speakers": 3,
            "balance_index": 0.185,
            "chance_acc": 1 / 3,
        },
    ]


def table_ii_results() -> list[dict[str, Any]]:
    """Table 2 — VLAAI decoding under LOTO / LOPEO / LOEO (paper lines 1–28)."""
    return [
        {"line": 1, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 0.0, "loss": "PCC", "acc": 0.6493, "acc_std": 0.0156, "rho_a": 0.0984, "rho_u": 0.0499, "rho_delta": 0.0485},
        {"line": 2, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 0.6, "loss": "PCC", "acc": 0.6895, "acc_std": 0.0065, "rho_a": 0.1124, "rho_u": 0.0488, "rho_delta": 0.0636},
        {"line": 3, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 1.0, "loss": "PCC", "acc": 0.8319, "acc_std": 0.0101, "rho_a": 0.1335, "rho_u": 0.0153, "rho_delta": 0.1182},
        {"line": 4, "cv": "LOTO", "dataset": "DTU", "chance": 0.5, "bi": 0.056, "loss": "PCC", "acc": 0.6527, "acc_std": 0.0206, "rho_a": 0.1170, "rho_u": 0.0272, "rho_delta": 0.0899},
        {"line": 5, "cv": "LOTO", "dataset": "DTU", "chance": 0.5, "bi": 1.0, "loss": "PCC", "acc": 0.6823, "acc_std": 0.0269, "rho_a": 0.1233, "rho_u": 0.0168, "rho_delta": 0.1066},
        {"line": 6, "cv": "LOTO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 0.185, "loss": "PCC", "acc": 0.3398, "acc_std": 0.0086, "rho_a": 0.0748, "rho_u": 0.0706, "rho_delta": 0.0043},
        {"line": 7, "cv": "LOTO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 1.0, "loss": "PCC", "acc": 0.5951, "acc_std": 0.0378, "rho_a": 0.1146, "rho_u": 0.0431, "rho_delta": 0.0714},
        {"line": 8, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 0.0, "loss": "PCC_delta", "acc": 0.6650, "acc_std": 0.0104, "rho_a": 0.0581, "rho_u": -0.0009, "rho_delta": 0.0589},
        {"line": 9, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 0.6, "loss": "PCC_delta", "acc": 0.7171, "acc_std": 0.0111, "rho_a": 0.0654, "rho_u": -0.0217, "rho_delta": 0.0871},
        {"line": 10, "cv": "LOTO", "dataset": "KUL", "chance": 0.5, "bi": 1.0, "loss": "PCC_delta", "acc": 0.8960, "acc_std": 0.0097, "rho_a": 0.0921, "rho_u": -0.0680, "rho_delta": 0.1601},
        {"line": 11, "cv": "LOTO", "dataset": "DTU", "chance": 0.5, "bi": 0.056, "loss": "PCC_delta", "acc": 0.5623, "acc_std": 0.0217, "rho_a": 0.0453, "rho_u": 0.0034, "rho_delta": 0.0419},
        {"line": 12, "cv": "LOTO", "dataset": "DTU", "chance": 0.5, "bi": 1.0, "loss": "PCC_delta", "acc": 0.6393, "acc_std": 0.0186, "rho_a": 0.0636, "rho_u": -0.0191, "rho_delta": 0.0827},
        {"line": 13, "cv": "LOTO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 0.185, "loss": "PCC_delta", "acc": 0.3730, "acc_std": 0.0044, "rho_a": 0.0137, "rho_u": 0.0020, "rho_delta": 0.0117},
        {"line": 14, "cv": "LOTO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 1.0, "loss": "PCC_delta", "acc": 0.6765, "acc_std": 0.0220, "rho_a": 0.0748, "rho_u": -0.0315, "rho_delta": 0.1063},
        {"line": 15, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 0.0, "loss": "PCC", "acc": 0.6493, "acc_std": 0.0156, "rho_a": 0.0984, "rho_u": 0.0499, "rho_delta": 0.0485},
        {"line": 16, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 0.6, "loss": "PCC", "acc": 0.6420, "acc_std": 0.0143, "rho_a": 0.0707, "rho_u": 0.0240, "rho_delta": 0.0467},
        {"line": 17, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 1.0, "loss": "PCC", "acc": 0.6467, "acc_std": 0.0512, "rho_a": 0.0599, "rho_u": 0.0139, "rho_delta": 0.0460},
        {"line": 18, "cv": "LOPEO", "dataset": "DTU", "chance": 0.5, "bi": 0.056, "loss": "PCC", "acc": 0.6662, "acc_std": 0.0143, "rho_a": 0.1130, "rho_u": 0.0204, "rho_delta": 0.0927},
        {"line": 19, "cv": "LOPEO", "dataset": "DTU", "chance": 0.5, "bi": 1.0, "loss": "PCC", "acc": 0.6334, "acc_std": 0.0443, "rho_a": 0.0944, "rho_u": 0.0120, "rho_delta": 0.0824},
        {"line": 20, "cv": "LOEO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 0.185, "loss": "PCC", "acc": 0.2792, "acc_std": 0.0213, "rho_a": 0.0424, "rho_u": 0.0557, "rho_delta": -0.0133},
        {"line": 21, "cv": "LOEO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 1.0, "loss": "PCC", "acc": 0.3442, "acc_std": 0.0677, "rho_a": 0.0331, "rho_u": 0.0255, "rho_delta": 0.0076},
        {"line": 22, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 0.0, "loss": "PCC_delta", "acc": 0.6688, "acc_std": 0.0141, "rho_a": 0.0564, "rho_u": -0.0019, "rho_delta": 0.0583},
        {"line": 23, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 0.6, "loss": "PCC_delta", "acc": 0.6399, "acc_std": 0.0411, "rho_a": 0.0424, "rho_u": -0.0038, "rho_delta": 0.0462},
        {"line": 24, "cv": "LOPEO", "dataset": "KUL", "chance": 0.5, "bi": 1.0, "loss": "PCC_delta", "acc": 0.6599, "acc_std": 0.0749, "rho_a": 0.0346, "rho_u": -0.0141, "rho_delta": 0.0487},
        {"line": 25, "cv": "LOPEO", "dataset": "DTU", "chance": 0.5, "bi": 0.056, "loss": "PCC_delta", "acc": 0.5719, "acc_std": 0.0300, "rho_a": 0.0517, "rho_u": 0.0038, "rho_delta": 0.0479},
        {"line": 26, "cv": "LOPEO", "dataset": "DTU", "chance": 0.5, "bi": 1.0, "loss": "PCC_delta", "acc": 0.5784, "acc_std": 0.0196, "rho_a": 0.0481, "rho_u": 0.0021, "rho_delta": 0.0460},
        {"line": 27, "cv": "LOEO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 0.185, "loss": "PCC_delta", "acc": 0.2391, "acc_std": 0.0233, "rho_a": -0.0314, "rho_u": -0.0021, "rho_delta": -0.0294},
        {"line": 28, "cv": "LOEO", "dataset": "NJU cEEGrid", "chance": 0.333, "bi": 1.0, "loss": "PCC_delta", "acc": 0.4451, "acc_std": 0.0781, "rho_a": 0.0023, "rho_u": -0.0268, "rho_delta": 0.0292},
    ]


def headline_results() -> dict[str, Any]:
    rows = table_ii_results()
    kul_loto = [r for r in rows if r["dataset"] == "KUL" and r["cv"] == "LOTO" and r["loss"] == "PCC"]
    kul_lopeo = [r for r in rows if r["dataset"] == "KUL" and r["cv"] == "LOPEO" and r["loss"] == "PCC"]
    return {
        "finding": (
            "Under LOTO, KUL accuracy rises from 0.65 (BI=0) to 0.83 (BI=1); "
            "under LOPEO all three KUL BI settings stay ~0.64–0.65."
        ),
        "kul_loto_acc_by_bi": {r["bi"]: r["acc"] for r in kul_loto},
        "kul_lopeo_acc_by_bi": {r["bi"]: r["acc"] for r in kul_lopeo},
        "significance": {
            "KUL_BI0_vs_BI1_LOTO": "p < 0.01",
            "NJU_BI0_vs_BI1_LOTO": "p < 0.001",
            "DTU_BI0_vs_BI1_LOTO": "n.s. (p = 0.12 PCC; p = 0.08 contrastive)",
        },
    }


def filter_table_ii(
    *,
    cv: str | None = None,
    dataset: str | None = None,
    loss: str | None = None,
) -> list[dict[str, Any]]:
    rows = table_ii_results()
    out = rows
    if cv is not None:
        out = [r for r in out if r["cv"] == cv]
    if dataset is not None:
        out = [r for r in out if r["dataset"] == dataset]
    if loss is not None:
        out = [r for r in out if r["loss"] == loss]
    return out


def evaluation_demo() -> dict[str, Any]:
    """Toy BI, metrics, and CV leakage check (no EEG / VLAAI)."""
    rng = np.random.default_rng(42)
    balanced_trials = [
        {"trial_id": f"b{i}", "attended": "A", "unattended": "B"}
        if i % 2 == 0
        else {"trial_id": f"b{i}", "attended": "B", "unattended": "A"}
        for i in range(8)
    ]
    unbalanced_trials = [
        {"trial_id": f"u{i}", "attended": "A", "unattended": "B"} for i in range(8)
    ]
    att_b, unatt_b = counts_from_trials(balanced_trials)
    att_u, unatt_u = counts_from_trials(unbalanced_trials)
    bi_balanced = balance_index(att_b, unatt_b)
    bi_unbalanced = balance_index(att_u, unatt_u)

    y_att = rng.standard_normal(64)
    y_unatt = rng.standard_normal(64)
    y_hat = y_att + 0.05 * rng.standard_normal(64)
    rho_a = pearson_corr(y_hat, y_att)
    rho_u = pearson_corr(y_hat, y_unatt)
    acc = decoding_accuracy([rho_a], [rho_u])

    loto_fold = partition_trials_loto(unbalanced_trials)[0]
    lopeo_fold = partition_trials_lopeo(unbalanced_trials, k=2)[0]
    loto_leak = pair_leakage_count(loto_fold[0], loto_fold[2])
    lopeo_leak = pair_leakage_count(lopeo_fold[0], lopeo_fold[2])

    # Memorization-style mock: identity feature boosts rho when pair seen in train.
    def mock_acc(train: list[dict], test: list[dict], leak_sensitive: bool) -> float:
        train_pairs = {(t["attended"], t["unattended"]) for t in train}
        rhos_a, rhos_u = [], []
        for t in test:
            boost = 0.15 if leak_sensitive and (t["attended"], t["unattended"]) in train_pairs else 0.0
            rhos_a.append(0.12 + boost + 0.01 * rng.random())
            rhos_u.append(0.05 + 0.01 * rng.random())
        return decoding_accuracy(rhos_a, rhos_u)

    kul_bal = construct_kul_trials(balanced=True)
    kul_unbal = construct_kul_trials(balanced=False)
    decoder = IdentityMemorizingDecoder()
    loto_train, _, loto_test = partition_trials_loto(kul_unbal)[0]
    decoder.fit(loto_train)
    kul_unbal_loto_acc = decoder.evaluate(loto_test)

    return {
        "kul_constructed_bi_balanced": dataset_condition_summary(kul_bal)["balance_index"],
        "kul_constructed_bi_unbalanced": dataset_condition_summary(kul_unbal)["balance_index"],
        "balance_index_balanced": bi_balanced,
        "balance_index_unbalanced": bi_unbalanced,
        "envelope_reconstruction": envelope_reconstruction_demo(),
        "identity_decoder_kul_unbal_loto_acc": kul_unbal_loto_acc,
        "pearson_attended": rho_a,
        "pearson_unattended": rho_u,
        "rho_delta": rho_delta(rho_a, rho_u),
        "decoding_accuracy_toy": acc,
        "pcc_loss": pcc_loss(rho_a),
        "contrastive_pcc_loss": contrastive_pcc_loss(rho_a, [rho_u]),
        "loto_pair_leakage": loto_leak,
        "lopeo_pair_leakage": lopeo_leak,
        "mock_acc_loto_leaky": mock_acc(loto_fold[0], loto_fold[2], leak_sensitive=True),
        "mock_acc_lopeo_strict": mock_acc(lopeo_fold[0], lopeo_fold[2], leak_sensitive=False),
        "loeo_partitions": len(partition_trials_loeo(unbalanced_trials, k=2)),
    }


def benchmarks_bundle() -> dict[str, Any]:
    kul_bal = construct_kul_trials(balanced=True)
    kul_unbal = construct_kul_trials(balanced=False)
    return {
        "limitations": LIMITATIONS,
        "framework": framework_card(),
        "table_i_datasets": table_i_datasets(),
        "table_ii_results": table_ii_results(),
        "headlines": headline_results(),
        "significance": significance_table(),
        "lopeo_mitigation": lopeo_mitigation_summary(),
        "constructed_conditions": {
            "KUL_balanced": dataset_condition_summary(kul_bal),
            "KUL_unbalanced": dataset_condition_summary(kul_unbal),
            "DTU_unbalanced": dataset_condition_summary(construct_dtu_trials(balanced=False)),
            "NJU_unbalanced": dataset_condition_summary(construct_nju_ceegrid_trials(balanced=False)),
        },
    }


def pipeline_demo() -> dict[str, Any]:
    return {
        "framework": framework_card(),
        "evaluation": evaluation_demo(),
        "headlines": headline_results(),
    }
