"""Leave-one-device-out federated evaluation (BTS-CAFE Table 2)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.btscafe.config import BTSCafeConfig
from ltx_trainer.btscafe.devices import device_registry, lodo_splits
from ltx_trainer.btscafe.federated import fedavg_aggregate, gradient_alignment_smoke, local_loss


def _device_features(device: dict[str, Any], *, dim: int = 16, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + hash(device["device"]) % 2**31)
    base = np.array([device["mean_mu_d"], device["mean_sf"], device["mean_st"]], dtype=np.float64)
    base = (base - base.mean()) / (base.std() + 1e-8)
    return np.concatenate([base, rng.standard_normal(max(0, dim - 3))])


def run_lodo_fold(
    held_out: str,
    *,
    train_devices: list[str],
    cfg: BTSCafeConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = cfg or BTSCafeConfig()
    reg = {d["device"]: d for d in device_registry()}
    train_vecs = [_device_features(reg[name], seed=seed) for name in train_devices if name in reg]
    test_vec = _device_features(reg[held_out], seed=seed + 1)
    if not train_vecs:
        return {"held_out": held_out, "score": 0.0}
    agg = fedavg_aggregate([1.0] * len(train_vecs), train_vecs)
    # cosine similarity proxy for held-out generalization
    score = float(np.dot(agg, test_vec) / (np.linalg.norm(agg) * np.linalg.norm(test_vec) + 1e-8))
    align = gradient_alignment_smoke(num_clients=len(train_vecs), param_dim=16, seed=seed)
    loss = local_loss(0.8, 0.6, align["mean_align_penalty"], round_idx=cfg.t_w + 1, cfg=cfg)
    return {"held_out": held_out, "score": score, "local_loss": loss["loss"], "alignment_active": loss["alignment_active"]}


def run_all_lodo(*, cfg: BTSCafeConfig | None = None, seed: int = 42) -> dict[str, Any]:
    results = []
    for split in lodo_splits():
        if isinstance(split["held_out"], list):
            continue
        results.append(
            run_lodo_fold(
                split["held_out"],
                train_devices=list(split["train_devices"]),
                cfg=cfg,
                seed=seed,
            )
        )
    scores = [r["score"] for r in results]
    return {"folds": results, "mean_score": float(np.mean(scores)), "n_folds": len(results)}


def lodo_eval_smoke(cfg: BTSCafeConfig | None = None) -> dict[str, Any]:
    out = run_all_lodo(cfg=cfg)
    return {"mean_score_finite": np.isfinite(out["mean_score"]), **out}
