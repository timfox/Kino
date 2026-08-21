"""Framework card, demos, and smoke (arXiv:2606.06418)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dopr.config import DoPrConfig, SyntheticFeatureConfig
from ltx_trainer.dopr.feature_learning import run_feature_learning_synthetic
from ltx_trainer.dopr.invariance import run_affine_invariance_demo
from ltx_trainer.dopr.metrics import (
    operating_guidelines,
    table_gsm8k_sft_3b,
    table_gsm8k_sft_8b_lr_sweep,
    table_humanoid_terminal_reward,
    table_robomimic_success,
    table_sit_fid_imagenet256,
)
from ltx_trainer.dopr.preconditioning import activation_precondition
from ltx_trainer.dopr.ttf import paper_mismatch_construction
import numpy as np


def framework_card(cfg: DoPrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DoPrConfig()
    return {
        "name": "DoPr",
        "paper": cfg.paper_arxiv,
        "title": "Double Preconditioning for test-time performance",
        "problem": "Test-time feedback (TTF): validation loss ≠ rollout reward",
        "recipe": "AP (activation Σ_z^{-1}) then GP (Adam/Muon/Signum)",
        "gp_options": ["sgd", "adam", "adamw", "muon", "signum", "adamuon"],
        "packages": list(cfg.packages),
        "domains": ["behavior_cloning", "autoregressive_lm", "flow_generative"],
        "code": "https://tinyurl.com/3kfuhmpf",
    }


def paper_limitations() -> list[str]:
    return [
        "Full Cholesky AP is O(d_in^3) per layer; large vocab embeddings need diagonal shortcut only.",
        "Distributed FSDP sharding not implemented — DDP-only in upstream paper.",
        "Benchmark numbers are literature anchors, not reproduced training runs in this stub.",
        "Conv SUA and rank-1+diag approximations are simplified versus full KFAC.",
        "Dropout + AP EMA interaction documented but not fully modeled here.",
    ]


def evaluation_demo(cfg: DoPrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DoPrConfig()
    feat_cfg = SyntheticFeatureConfig(steps=80, seed=7)
    sgd = run_feature_learning_synthetic("sgd", feat_cfg)
    adam = run_feature_learning_synthetic("adam", feat_cfg)
    dopr = run_feature_learning_synthetic("dopr_sgd", feat_cfg)
    inv_dopr = run_affine_invariance_demo(use_dopr=True, steps=25)
    inv_sgd = run_affine_invariance_demo(use_dopr=False, steps=25)
    ttf = paper_mismatch_construction()

    rng = np.random.RandomState(0)
    z = rng.randn(64, 8)
    g = rng.randn(4, 8)
    m = activation_precondition(g, z, damping=1e-3)

    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "guidelines": operating_guidelines(),
        "ap_shape": {"gradient": list(g.shape), "preconditioned": list(m.shape)},
        "feature_learning": {
            "sgd_final_dist": sgd.subspace_distance[-1],
            "adam_final_dist": adam.subspace_distance[-1],
            "dopr_sgd_final_dist": dopr.subspace_distance[-1],
            "sgd_final_loss": sgd.validation_loss[-1],
            "dopr_sgd_final_loss": dopr.validation_loss[-1],
        },
        "invariance": {
            "dopr_max_loss_diff": inv_dopr.max_abs_diff,
            "sgd_max_loss_diff": inv_sgd.max_abs_diff,
        },
        "ttf_mismatch": ttf.__dict__,
        "tables": {
            "gsm8k_3b": table_gsm8k_sft_3b(),
            "gsm8k_8b_sweep": table_gsm8k_sft_8b_lr_sweep(),
            "humanoid": table_humanoid_terminal_reward(),
            "robomimic": table_robomimic_success(),
            "sit_fid": table_sit_fid_imagenet256(),
        },
    }


def evaluation_smoke(cfg: DoPrConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    fl = demo["feature_learning"]
    assert fl["dopr_sgd_final_dist"] < fl["sgd_final_dist"]
    assert demo["invariance"]["dopr_max_loss_diff"] < demo["invariance"]["sgd_max_loss_diff"]
    ttf = demo["ttf_mismatch"]
    assert ttf["lval_pi1"] < ttf["lval_pi2"]
    assert ttf["lideal_pi2"] < ttf["lideal_pi1"]
    return {"status": "ok", "paper": (cfg or DoPrConfig()).paper_arxiv, "demo_keys": list(demo.keys())}
