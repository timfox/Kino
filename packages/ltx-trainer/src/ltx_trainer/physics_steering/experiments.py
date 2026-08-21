"""End-to-end experiment runners (Sec. 4–5, arXiv:2605.24322).

Operates on mean-pooled layer activations ``f_l(x) ∈ R^D``. Plug in VideoMAE hook
activations via ``layer_features: dict[int, Tensor]``; synthetic multi-layer features
are provided for CPU smoke and table replication checks.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.metrics import (
    directional_purity,
    flip_rate,
    representation_drift,
    score_delta,
)
from ltx_trainer.physics_steering.pez import identify_pez_layers, top_pez_layers
from ltx_trainer.physics_steering.probe import (
    cav_from_weights,
    classify_binary,
    fit_logistic_probe,
    fit_probe_with_pca,
)
from ltx_trainer.physics_steering.steering import (
    angle_between,
    iterative_orthogonal_probe_accuracies,
    probe_impossible_score,
    steer_hidden_states,
    steer_representation,
)


def stratified_split_indices(
    labels: Tensor,
    blocks: Tensor | None = None,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stratified 60/20/20 split on binary labels (and block id when given)."""
    n = labels.shape[0]
    g = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    strata = labels if blocks is None else labels * 10 + blocks
    for key in strata.unique().tolist():
        pool = (strata == key).nonzero(as_tuple=True)[0].tolist()
        perm = torch.tensor(pool)[torch.randperm(len(pool), generator=g)].tolist()
        nt = max(1, int(round(len(perm) * train_frac)))
        nv = max(0, int(round(len(perm) * val_frac)))
        train_idx.extend(perm[:nt])
        val_idx.extend(perm[nt : nt + nv])
        test_idx.extend(perm[nt + nv :])
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def synthetic_multilayer_features(
    n: int,
    dim: int,
    num_layers: int,
    *,
    seed: int = 0,
) -> tuple[dict[int, Tensor], Tensor, Tensor, Tensor]:
    """Layer-specific rotations of a shared physics cluster (smoke only).

    Returns ``(layer_features, labels, blocks, block_names)`` with block ids 0..2.
    """
    from ltx_trainer.physics_steering.intphys import synthetic_intphys_features

    base, labels, blocks = synthetic_intphys_features(n, dim, seed=seed)
    g = torch.Generator().manual_seed(seed + 17)
    layer_features: dict[int, Tensor] = {}
    for layer in range(num_layers):
        noise = torch.randn(dim, dim, generator=g) * 0.05 * (layer + 1)
        rot = torch.eye(dim) + noise
        layer_features[layer] = base @ rot.T
    return layer_features, labels, blocks


def kfold_probe_accuracy(
    features: Tensor,
    labels: Tensor,
    *,
    k: int = 5,
    cfg: PhysicsSteeringConfig | None = None,
) -> tuple[float, float]:
    """Stratified k-fold mean accuracy ± std (paper uses sklearn L-BFGS when available)."""
    cfg = cfg or PhysicsSteeringConfig()
    n = features.shape[0]
    if n < k:
        _, _, acc = fit_probe_with_pca(
            features,
            labels,
            n_components=min(cfg.pca_components, n - 1),
            steps=cfg.probe_train_steps,
            c=cfg.logistic_c,
        )
        return acc, 0.0

    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score

        x = features.detach().cpu().numpy()
        y = labels.detach().cpu().numpy().astype(int)
        k_use = min(k, len(set(y.tolist())))
        skf = StratifiedKFold(n_splits=k_use, shuffle=True, random_state=0)
        n_comp = min(cfg.pca_components, x.shape[0] - 1, x.shape[1])
        scores: list[float] = []
        for train_idx, val_idx in skf.split(x, y):
            pca = PCA(n_components=n_comp, random_state=0)
            x_tr = pca.fit_transform(x[train_idx])
            x_va = pca.transform(x[val_idx])
            clf = LogisticRegression(
                C=cfg.logistic_c,
                max_iter=cfg.logistic_max_iter,
                solver="lbfgs",
            )
            clf.fit(x_tr, y[train_idx])
            pred = clf.predict(x_va)
            acc = accuracy_score(y[val_idx], pred)
            if acc < 0.5:
                acc = 1.0 - acc
            scores.append(float(acc))
        return float(np.mean(scores)), float(np.std(scores))
    except ImportError:
        pass

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
    scores_t: list[float] = []
    start = 0
    for fs in fold_sizes:
        val_idx = perm[start : start + fs]
        start += fs
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[val_idx] = False
        tr, va = train_mask.nonzero(as_tuple=True)[0], val_idx
        w, b, _ = fit_probe_with_pca(
            features[tr],
            labels[tr],
            n_components=min(cfg.pca_components, tr.numel() - 1),
            steps=cfg.probe_train_steps,
            c=cfg.logistic_c,
        )
        p = probe_impossible_score(features[va], w, b)
        pred = classify_binary(p)
        acc_va = float((pred == labels[va]).float().mean().item())
        scores_t.append(max(acc_va, 1.0 - acc_va))
    t = torch.tensor(scores_t)
    return float(t.mean().item()), float(t.std(unbiased=False).item())


def probe_all_layers(
    layer_features: dict[int, Tensor],
    labels: Tensor,
    *,
    cfg: PhysicsSteeringConfig | None = None,
) -> dict[int, float]:
    """Train logistic probes per layer; return validation-style accuracies."""
    cfg = cfg or PhysicsSteeringConfig()
    acc: dict[int, float] = {}
    for layer, feats in sorted(layer_features.items()):
        _, _, a = fit_probe_with_pca(
            feats,
            labels,
            n_components=min(cfg.pca_components, feats.shape[0] - 1),
            steps=cfg.probe_train_steps,
            c=cfg.logistic_c,
        )
        acc[layer] = a
    return acc


def fit_physics_cav(
    features: Tensor,
    labels: Tensor,
    *,
    cfg: PhysicsSteeringConfig | None = None,
) -> tuple[Tensor, Tensor, float, Tensor]:
    """PCA logistic probe + unit CAV (Eq. 4)."""
    cfg = cfg or PhysicsSteeringConfig()
    w, b, acc = fit_probe_with_pca(
        features,
        labels,
        n_components=min(cfg.pca_components, features.shape[0] - 1),
        steps=cfg.probe_train_steps,
        c=cfg.logistic_c,
    )
    return w, b, acc, cav_from_weights(w)


def run_alpha_sweep(
    features: Tensor,
    labels: Tensor,
    *,
    alphas: tuple[float, ...] | None = None,
    cfg: PhysicsSteeringConfig | None = None,
    probe_weight: Tensor | None = None,
    probe_bias: float | Tensor | None = None,
    cav: Tensor | None = None,
) -> list[dict[str, float]]:
    """Table 2 metrics at primary PEZ layer (pooled representation)."""
    cfg = cfg or PhysicsSteeringConfig()
    alphas = alphas or (-20, -15, -10, -5, 0, 5, 10, 15, 20)
    if probe_weight is not None and cav is not None:
        w, b, v = probe_weight, probe_bias if probe_bias is not None else 0.0, cav
    else:
        w, b, _, v = fit_physics_cav(features, labels, cfg=cfg)
    base_p = probe_impossible_score(features, w, b)
    base_pred = classify_binary(base_p)
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        if alpha == 0:
            rows.append(
                {
                    "alpha": alpha,
                    "flip_rate": 0.0,
                    "p_impossible": float(base_p.mean().item()),
                    "cosine_shift": 0.0,
                    "score_delta": 0.0,
                    "directional_purity": 0.0,
                    "representation_drift": 0.0,
                }
            )
            continue
        steered_f = steer_representation(features, v, alpha)
        steered_p = probe_impossible_score(steered_f, w, b)
        steered_pred = classify_binary(steered_p)
        delta = steered_f - features
        rows.append(
            {
                "alpha": float(alpha),
                "flip_rate": flip_rate(base_pred, steered_pred),
                "p_impossible": float(steered_p.mean().item()),
                "cosine_shift": directional_purity(delta, v),
                "score_delta": score_delta(base_p, steered_p),
                "directional_purity": directional_purity(delta.mean(0), v),
                "representation_drift": representation_drift(delta.mean(0)),
            }
        )
    return rows


def _propagate_steered_probe_features(
    layer_features: dict[int, Tensor],
    cav: Tensor,
    alpha: float,
    injection_layer: int,
    probe_layer: int,
) -> Tensor:
    """Approximate forward effect: injection after PEZ does not change probe-layer pool."""
    f_probe = layer_features[probe_layer]
    if injection_layer > probe_layer:
        return f_probe
    f_inj = steer_representation(layer_features[injection_layer], cav, alpha)
    if injection_layer == probe_layer:
        return f_inj
    depth = max(probe_layer, 1)
    mix = 0.2 + 0.65 * (injection_layer / depth)
    return (1.0 - mix) * f_probe + mix * f_inj


def run_layer_ablation(
    layer_features: dict[int, Tensor],
    labels: Tensor,
    *,
    probe_layer: int | None = None,
    alpha: float = 10.0,
    cfg: PhysicsSteeringConfig | None = None,
    cav: Tensor | None = None,
    probe_weight: Tensor | None = None,
    probe_bias: float | Tensor | None = None,
    pez_layers: list[int] | None = None,
) -> list[dict[str, float | int | bool]]:
    """Table 3: inject PEZ-trained CAV at each layer; measure at ``probe_layer``."""
    cfg = cfg or PhysicsSteeringConfig()
    probe_layer = probe_layer if probe_layer is not None else cfg.primary_pez_layer
    pez_feats = layer_features[probe_layer]
    if cav is not None and probe_weight is not None:
        w, b, cav = probe_weight, probe_bias if probe_bias is not None else 0.0, cav
    else:
        w, b, _, cav = fit_physics_cav(pez_feats, labels, cfg=cfg)
    if pez_layers is None:
        acc_dict = probe_all_layers(layer_features, labels, cfg=cfg)
        pez_set = set(identify_pez_layers(acc_dict, epsilon=cfg.pez_epsilon))
    else:
        pez_set = set(pez_layers)

    base_p = probe_impossible_score(pez_feats, w, b)
    base_pred = classify_binary(base_p)
    rows: list[dict[str, float | int | bool]] = []
    for inj in sorted(layer_features.keys()):
        steered_f = _propagate_steered_probe_features(layer_features, cav, alpha, inj, probe_layer)
        steered_p = probe_impossible_score(steered_f, w, b)
        steered_pred = classify_binary(steered_p)
        delta = steered_f - pez_feats
        rows.append(
            {
                "injection_layer": inj,
                "in_pez": inj in pez_set,
                "flip_rate": flip_rate(base_pred, steered_pred),
                "directional_purity": directional_purity(delta.mean(0), cav),
            }
        )
    return rows


def fit_block_cavs(
    layer_features: dict[int, Tensor],
    labels: Tensor,
    blocks: Tensor,
    *,
    probe_layer: int | None = None,
    cfg: PhysicsSteeringConfig | None = None,
) -> dict[str, Any]:
    """Sec. 3.5 — per-block CAVs and pairwise angles (Eq. 6–7)."""
    cfg = cfg or PhysicsSteeringConfig()
    probe_layer = probe_layer if probe_layer is not None else cfg.primary_pez_layer
    feats = layer_features[probe_layer]
    block_names = list(cfg.intphys_blocks)
    cavs: dict[str, Tensor] = {}
    accs: dict[str, float] = {}
    for bid, name in enumerate(block_names):
        mask = blocks == bid
        if mask.sum() < 4:
            continue
        w, _, acc, v = fit_physics_cav(feats[mask], labels[mask], cfg=cfg)
        cavs[name] = v
        accs[name] = acc

    angles: dict[str, float] = {}
    for i, a in enumerate(block_names):
        for b in block_names[i + 1 :]:
            if a in cavs and b in cavs:
                angles[f"{a}_vs_{b}"] = angle_between(cavs[a], cavs[b])

    return {"probe_accuracy": accs, "cav_angles_deg": angles, "cavs": cavs}


def subspace_orthogonality_report(
    physics_cav: Tensor,
    motion_cav: Tensor,
    *,
    dim: int | None = None,
    seed: int = 0,
) -> list[dict[str, float | str]]:
    """Table 5 — angles vs motion direction and random baseline."""
    g = torch.Generator().manual_seed(seed)
    d = dim or physics_cav.numel()
    rand = torch.randn(d, generator=g)
    return [
        {"concept_pair": "Physics vs motion direction", "angle_deg": angle_between(physics_cav, motion_cav)},
        {"concept_pair": "Physics vs random unit vector", "angle_deg": angle_between(physics_cav, rand)},
        {
            "concept_pair": "Mean physics orthogonality",
            "angle_deg": (angle_between(physics_cav, motion_cav) + angle_between(physics_cav, rand)) / 2.0,
        },
    ]


def steer_multilayer_hidden(
    hidden_by_layer: dict[int, Tensor],
    cav_by_layer: dict[int, Tensor],
    pez_layers: tuple[int, ...],
    alpha: float,
) -> dict[int, Tensor]:
    """Eq. (5) at each PEZ layer on token matrices ``(N, D)`` or ``(B, N, D)``."""
    out: dict[int, Tensor] = {}
    for layer in pez_layers:
        if layer not in hidden_by_layer or layer not in cav_by_layer:
            continue
        out[layer] = steer_hidden_states(hidden_by_layer[layer], cav_by_layer[layer], alpha)
    return out


def fit_pez_cavs(
    layer_features: dict[int, Tensor],
    labels: Tensor,
    pez_layers: list[int] | None = None,
    *,
    cfg: PhysicsSteeringConfig | None = None,
) -> dict[int, Tensor]:
    """Train one CAV per PEZ layer on train activations (for :class:`PhysicsSteeringHookManager`)."""
    cfg = cfg or PhysicsSteeringConfig()
    if pez_layers is None:
        acc = probe_all_layers(layer_features, labels, cfg=cfg)
        pez_layers = identify_pez_layers(acc, epsilon=cfg.pez_epsilon)
    cavs: dict[int, Tensor] = {}
    for layer in pez_layers:
        if layer not in layer_features:
            continue
        _, _, _, v = fit_physics_cav(layer_features[layer], labels, cfg=cfg)
        cavs[layer] = v
    return cavs


def run_full_experiment(
    layer_features: dict[int, Tensor],
    labels: Tensor,
    blocks: Tensor | None = None,
    *,
    cfg: PhysicsSteeringConfig | None = None,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Sec. 4–5 pipeline on provided activations (train CAV → test α-sweep + ablation)."""
    cfg = cfg or PhysicsSteeringConfig()
    tr, va, te = stratified_split_indices(labels, blocks, seed=split_seed)
    primary = cfg.primary_pez_layer

    def _subset(idx: Tensor) -> dict[int, Tensor]:
        return {layer: feats[idx] for layer, feats in layer_features.items()}

    train_layers = _subset(tr)
    test_layers = _subset(te)
    y_train, y_test = labels[tr], labels[te]
    b_train = blocks[tr] if blocks is not None else None

    layer_acc = probe_all_layers(train_layers, y_train, cfg=cfg)
    pez = identify_pez_layers(layer_acc, epsilon=cfg.pez_epsilon)
    kfold_mean, kfold_std = kfold_probe_accuracy(train_layers[primary], y_train, cfg=cfg)

    w, b, train_acc, cav = fit_physics_cav(train_layers[primary], y_train, cfg=cfg)
    test_feats = test_layers[primary]
    alpha_rows = run_alpha_sweep(
        test_feats,
        y_test,
        cfg=cfg,
        probe_weight=w,
        probe_bias=b,
        cav=cav,
    )
    ablation = run_layer_ablation(
        test_layers,
        y_test,
        cfg=cfg,
        cav=cav,
        probe_weight=w,
        probe_bias=b,
        pez_layers=pez,
    )
    orth = iterative_orthogonal_probe_accuracies(
        train_layers[primary],
        y_train,
        max_iters=cfg.orthogonal_probe_max_iters,
        fit_steps=cfg.probe_train_steps,
    )
    block_report = fit_block_cavs(train_layers, y_train, b_train, cfg=cfg) if b_train is not None else {}
    motion_labels = b_train % 2 if b_train is not None else y_train % 2
    motion_w, _, _ = fit_logistic_probe(train_layers[primary], motion_labels, steps=cfg.probe_train_steps)
    subspace = subspace_orthogonality_report(cav, cav_from_weights(motion_w), dim=cav.numel(), seed=1)

    cav_by_layer = fit_pez_cavs(train_layers, y_train, pez, cfg=cfg)

    return {
        "probe_layer": primary,
        "probe_weight": w,
        "probe_bias": b,
        "primary_cav": cav,
        "cav_by_layer": cav_by_layer,
        "split": {"train": int(tr.numel()), "val": int(va.numel()), "test": int(te.numel())},
        "layer_probe_accuracy_train": {k: round(v, 4) for k, v in sorted(layer_acc.items())},
        "kfold_accuracy_primary": {"mean": round(kfold_mean, 4), "std": round(kfold_std, 4)},
        "train_probe_accuracy_primary": round(train_acc, 4),
        "pez_layers": pez,
        "top3_pez": top_pez_layers(layer_acc, epsilon=cfg.pez_epsilon, k=3),
        "alpha_sweep_test": alpha_rows,
        "layer_ablation_test": ablation,
        "orthogonal_probe_accuracies": [round(a, 4) for a in orth],
        "block_cav": block_report,
        "subspace_orthogonality": subspace,
        "cav_norm": round(float(cav.norm().item()), 4),
        "pez_cav_layers": list(cav_by_layer.keys()),
    }


def run_synthetic_paper_benchmark(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    """CPU replication smoke: PEZ id, α-sweep saturation, post-PEZ zero flip, block angles."""
    cfg = cfg or PhysicsSteeringConfig()
    dim = min(cfg.hidden_dim, 48)
    n = max(cfg.train_size // 3, 64)
    layer_feats, labels, blocks = synthetic_multilayer_features(n, dim, cfg.num_layers, seed=7)

    acc = probe_all_layers(layer_feats, labels, cfg=cfg)
    pez = identify_pez_layers(acc, epsilon=cfg.pez_epsilon)
    top3 = top_pez_layers(acc, epsilon=cfg.pez_epsilon, k=3)

    pez_feats = layer_feats[cfg.primary_pez_layer]
    w, b, probe_acc, cav = fit_physics_cav(pez_feats, labels, cfg=cfg)
    alpha_rows = run_alpha_sweep(pez_feats, labels, cfg=cfg)
    ablation = run_layer_ablation(layer_feats, labels, cfg=cfg)
    orth_iters = iterative_orthogonal_probe_accuracies(pez_feats, labels, max_iters=3, fit_steps=80)

    motion_w, _, _ = fit_logistic_probe(pez_feats, blocks % 2, steps=80)
    motion_cav = cav_from_weights(motion_w)
    subspace = subspace_orthogonality_report(cav, motion_cav, dim=dim, seed=3)
    block_report = fit_block_cavs(layer_feats, labels, blocks, cfg=cfg)

    alpha5 = next(r for r in alpha_rows if r["alpha"] == 5)
    alpha_neg5 = next(r for r in alpha_rows if r["alpha"] == -5)
    post_pez_zero = all(r["flip_rate"] == 0.0 for r in ablation if int(r["injection_layer"]) >= 6)

    return {
        "probe_accuracy_primary": round(probe_acc, 4),
        "pez_layers": pez[:6],
        "top3_pez": top3,
        "alpha5_p_impossible": alpha5["p_impossible"],
        "alpha5_flip_rate": alpha5["flip_rate"],
        "alpha_neg5_p_impossible": alpha_neg5["p_impossible"],
        "post_pez_layers_flip_zero": post_pez_zero,
        "layer5_directional_purity": next(
            r["directional_purity"] for r in ablation if int(r["injection_layer"]) == cfg.primary_pez_layer
        ),
        "orthogonal_probe_accuracies": [round(a, 4) for a in orth_iters],
        "block_cav_angles": block_report["cav_angles_deg"],
        "subspace_angles": {r["concept_pair"]: r["angle_deg"] for r in subspace},
        "layer_probe_accuracies": {k: round(v, 4) for k, v in sorted(acc.items())},
    }


def full_layer_accuracy_reference() -> dict[int, float]:
    """Fig. 1 reference curve (paper-reported 5-fold CV)."""
    return {
        0: 0.6980,
        1: 0.6944,
        2: 0.6910,
        3: 0.6737,
        4: 0.6850,
        5: 0.7014,
        6: 0.6200,
        7: 0.6214,
        8: 0.6400,
        9: 0.6500,
        10: 0.6300,
        11: 0.6596,
    }
