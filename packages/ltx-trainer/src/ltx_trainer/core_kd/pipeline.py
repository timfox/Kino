"""CoRe-KD framework card, knowledge, and evaluation demos."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.core_kd.benchmarks import benchmarks_bundle
from ltx_trainer.core_kd.config import CoreKDConfig
from ltx_trainer.core_kd.csa import csa_loss
from ltx_trainer.core_kd.nce import build_conflict_view, mean_rejection_rate, nce_loss
from ltx_trainer.core_kd.poe import poe_fuse


def framework_card(cfg: CoreKDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CoreKDConfig()
    return {
        "name": "CoRe-KD",
        "paper": cfg.paper_arxiv,
        "title": "State-Anchored Complete-View Distillation for Robust Conversational MER",
        "components": ["CSA", "NCE"],
        "modalities": list(cfg.modalities),
        "state_dim": cfg.state_dim,
        "loss_weights": {
            "lambda_kd": cfg.lambda_kd,
            "lambda_state": cfg.lambda_state,
            "lambda_mstate": cfg.lambda_mstate,
            "lambda_nce": cfg.lambda_nce,
        },
    }


def knowledge_card(cfg: CoreKDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CoreKDConfig()
    bench = benchmarks_bundle()
    return {
        "framework": framework_card(cfg),
        "key_findings": {
            "iemocap6_lav_f1": bench["table1_iemocap6"]["{l,a,v}"][1],
            "state_drift_reduction_pct": bench["mechanism"]["state_drift_reduction_pct"],
            "nce_rejection_rate_pct": bench["mechanism"]["rejection_rate_core_kd_pct"],
        },
        "integration": {
            "env": "GOPEX_CORE_KD=1",
            "fold_hook": "GOPEX_AV_FOLD_HOOKS=...,core_kd",
            "cli": "./scripts/gopex-core-kd.sh",
        },
    }


def _toy_teacher_student(rng: np.random.Generator, d: int, n_cls: int) -> dict[str, Any]:
    mus = [rng.standard_normal(d) for _ in range(3)]
    precs = [np.ones(d) * (1.0 + 0.1 * i) for i in range(3)]
    t_mu, t_sig = poe_fuse(mus, precs)
    s_mu = t_mu + 0.05 * rng.standard_normal(d)
    s_sig = t_sig + 0.02 * rng.standard_normal(d)
    t_logits = rng.standard_normal(n_cls)
    s_logits = t_logits + 0.1 * rng.standard_normal(n_cls)
    modal = {f"m{i}": (mus[i], np.sqrt(1.0 / precs[i])) for i in range(3)}
    return {
        "teacher_fused": (t_mu, t_sig),
        "student_fused": (s_mu, s_sig),
        "teacher_logits": t_logits,
        "student_logits": s_logits,
        "teacher_modal": modal,
    }


def evaluation_demo(cfg: CoreKDConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CoreKDConfig()
    rng = np.random.default_rng(0)
    d, n_cls = 16, 6
    pack = _toy_teacher_student(rng, d, n_cls)
    y = 2
    unavailable = {"video"}
    pred_unavail = {
        "video": (
            pack["teacher_modal"]["m2"][0] + 0.1 * rng.standard_normal(d),
            pack["teacher_modal"]["m2"][1],
        )
    }
    csa = csa_loss(
        y,
        pack["student_logits"],
        pack["teacher_logits"],
        pack["student_fused"],
        pack["teacher_fused"],
        pred_unavail,
        {k: pack["teacher_modal"][k] for k in pack["teacher_modal"]},
        unavailable,
        cfg=cfg,
    )

    labels = np.array([0, 1, 2, 3, 2, 4])
    feat = {m: rng.standard_normal(d) for m in ("text", "audio", "video")}

    def conflict_logits_fn(i: int, j: int, replace: set[str]) -> np.ndarray:
        _ = build_conflict_view(feat, feat, replace)
        logits = rng.standard_normal(n_cls)
        logits[y] += 1.0
        return logits

    ln, n_nce = nce_loss(labels, conflict_logits_fn, rng=rng)
    pairs = [
        (pack["teacher_fused"][0], pack["teacher_modal"]["m1"][0], pack["student_fused"][0])
        for _ in range(8)
    ]
    rej = mean_rejection_rate(pairs)

    return {
        "csa": csa,
        "nce_loss": ln,
        "nce_samples": n_nce,
        "rejection_rate_stub": rej,
        "paper_tables": benchmarks_bundle(),
    }


def evaluation_smoke(cfg: CoreKDConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    bench = benchmarks_bundle()
    return {
        "ok": demo["csa"]["l_csa"] > 0,
        "l_csa": round(demo["csa"]["l_csa"], 4),
        "iemocap6_lav_f1": bench["table1_iemocap6"]["{l,a,v}"][1],
        "state_drift_reduction_pct": bench["mechanism"]["state_drift_reduction_pct"],
        "nce_rejection_pct": bench["mechanism"]["rejection_rate_core_kd_pct"],
    }
