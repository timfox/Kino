"""Synthetic representation transfer and safety-utility trade-off demos."""

from __future__ import annotations

import numpy as np

from ltx_trainer.cross_model_safety.alignment import fit_mlp_map, fit_ridge_map, fit_svd_map
from ltx_trainer.cross_model_safety.config import CrossModelSafetyConfig, SteeringParams
from ltx_trainer.cross_model_safety.metrics import (
    attack_success_rate,
    clip_similarity_proxy,
    unsafe_score_from_hidden,
)
from ltx_trainer.cross_model_safety.steering import (
    category_directions,
    source_safety_direction,
    steer_hidden,
    transfer_direction,
)


def _random_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def synthetic_setup(
    *,
    d_source: int = 32,
    d_target: int = 40,
    n_pairs: int = 64,
    n_anchors: int = 256,
    seed: int = 0,
) -> dict[str, object]:
    """Shared geometry: target ≈ W @ source + shared unsafe axis."""
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.4, size=(d_target, d_source))
    unsafe_s = rng.normal(size=d_source)
    unsafe_s /= np.linalg.norm(unsafe_s) + 1e-8
    unsafe_t = w @ unsafe_s
    unsafe_t /= np.linalg.norm(unsafe_t) + 1e-8
    caption_t = rng.normal(size=d_target)
    caption_t /= np.linalg.norm(caption_t) + 1e-8

    safe_s = rng.normal(size=(n_pairs, d_source))
    unsafe_samps = safe_s + 0.55 * unsafe_s
    v_source = source_safety_direction(safe_s, unsafe_samps)

    anchors_s = rng.normal(size=(d_source, n_anchors))
    anchors_t = w @ anchors_s + rng.normal(scale=0.05, size=(d_target, n_anchors))

    safe_t = (w @ safe_s.T).T + rng.normal(scale=0.04, size=(n_pairs, d_target))
    unsafe_t_samps = safe_t + 0.5 * unsafe_t

    by_cat_safe: dict[str, list[np.ndarray]] = {}
    by_cat_unsafe: dict[str, list[np.ndarray]] = {}
    cats = ["Sexual", "Violence", "Hate", "Illegal Activities"]
    for i in range(n_pairs):
        cat = cats[i % len(cats)]
        by_cat_safe.setdefault(cat, []).append(safe_s[i])
        by_cat_unsafe.setdefault(cat, []).append(unsafe_samps[i])

    safe_cat = {k: np.stack(v) for k, v in by_cat_safe.items()}
    unsafe_cat = {k: np.stack(v) for k, v in by_cat_unsafe.items()}

    return {
        "w_gt": w,
        "v_source": v_source,
        "unsafe_axis_target": unsafe_t,
        "caption_axis_target": caption_t,
        "anchors_source": anchors_s,
        "anchors_target": anchors_t,
        "safe_target": safe_t,
        "unsafe_target": unsafe_t_samps,
        "safe_source": safe_s,
        "unsafe_source": unsafe_samps,
        "category_safe_source": safe_cat,
        "category_unsafe_source": unsafe_cat,
    }


def fit_alignment(method: str, hs: np.ndarray, ht: np.ndarray, params: SteeringParams, rng: np.random.Generator):
    if method == "svd":
        return fit_svd_map(hs, ht)
    if method == "ridge":
        return fit_ridge_map(hs, ht, lam=params.ridge_lambda)
    return fit_mlp_map(hs, ht, hidden=min(params.mlp_hidden, hs.shape[0]), rng=rng)


def evaluate_transfer(
    setup: dict[str, object],
    method: str,
    *,
    alpha: float,
    params: SteeringParams | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    params = params or SteeringParams()
    rng = rng or np.random.default_rng(0)
    hs = setup["anchors_source"]  # type: ignore[index]
    ht = setup["anchors_target"]  # type: ignore[index]
    v_s = setup["v_source"]  # type: ignore[index]
    map_params = fit_alignment(method, hs, ht, params, rng)
    v_t = transfer_direction(v_s, hs, ht, method, map_params)

    unsafe_axis = setup["unsafe_axis_target"]  # type: ignore[index]
    caption_axis = setup["caption_axis_target"]  # type: ignore[index]
    safe_t = setup["unsafe_target"]  # type: ignore[index]  # evaluate on unsafe prompts

    scores = []
    clips = []
    for h in safe_t:
        h0 = h[np.newaxis, :] if h.ndim == 1 else h
        steered = steer_hidden(h0, v_t, alpha=alpha)
        scores.append(unsafe_score_from_hidden(steered, unsafe_axis))
        clips.append(clip_similarity_proxy(steered, caption_axis))

    return {
        "asr": attack_success_rate(np.array(scores)),
        "clip_sim": float(np.mean(clips)),
        "alpha": alpha,
        "method": method,
    }


def alpha_sweep_demo(
    setup: dict[str, object],
    method: str = "svd",
    *,
    alphas: tuple[float, ...] = (-1.0, 0.0, 1.0, 3.0, 5.0, 7.0),
) -> list[dict[str, float]]:
    return [evaluate_transfer(setup, method, alpha=a) for a in alphas]


def alignment_comparison(
    setup: dict[str, object],
    *,
    alpha: float = 5.0,
    params: SteeringParams | None = None,
) -> dict[str, dict[str, float]]:
    params = params or SteeringParams()
    rng = np.random.default_rng(1)
    out: dict[str, dict[str, float]] = {}
    for method in ("svd", "ridge", "mlp"):
        out[method] = evaluate_transfer(setup, method, alpha=alpha, params=params, rng=rng)
    return out


def multi_vector_demo(setup: dict[str, object], method: str = "ridge", *, alpha: float = 5.0) -> dict[str, float]:
    hs = setup["anchors_source"]  # type: ignore[index]
    ht = setup["anchors_target"]  # type: ignore[index]
    rng = np.random.default_rng(2)
    map_params = fit_alignment(method, hs, ht, SteeringParams(), rng)
    dirs_s = category_directions(
        setup["category_safe_source"],  # type: ignore[arg-type]
        setup["category_unsafe_source"],  # type: ignore[arg-type]
    )
    dirs_t = {
        c: transfer_direction(v, hs, ht, method, map_params)
        for c, v in dirs_s.items()
    }
    unsafe_axis = setup["unsafe_axis_target"]  # type: ignore[index]
    caption_axis = setup["caption_axis_target"]  # type: ignore[index]
    unsafe_t = setup["unsafe_target"]  # type: ignore[index]

    global_v = transfer_direction(setup["v_source"], hs, ht, method, map_params)  # type: ignore[arg-type]

    scores_global = []
    scores_multi = []
    clips_global = []
    clips_multi = []
    cats = list(dirs_t.keys())
    for i, h in enumerate(unsafe_t):
        cat = cats[i % len(cats)]
        h0 = h[np.newaxis, :]
        g = steer_hidden(h0, global_v, alpha=alpha)
        m = steer_hidden(h0, dirs_t[cat], alpha=alpha)
        scores_global.append(unsafe_score_from_hidden(g, unsafe_axis))
        scores_multi.append(unsafe_score_from_hidden(m, unsafe_axis))
        clips_global.append(clip_similarity_proxy(g, caption_axis))
        clips_multi.append(clip_similarity_proxy(m, caption_axis))

    return {
        "global_asr": attack_success_rate(np.array(scores_global)),
        "multi_asr": attack_success_rate(np.array(scores_multi)),
        "global_clip": float(np.mean(clips_global)),
        "multi_clip": float(np.mean(clips_multi)),
    }


def random_baseline_asr(
    setup: dict[str, object],
    *,
    alpha: float = 5.0,
    seed: int = 99,
    params: SteeringParams | None = None,
) -> float:
    """Random direction with the same magnitude as the transferred SVD vector."""
    params = params or SteeringParams()
    hs = setup["anchors_source"]  # type: ignore[index]
    ht = setup["anchors_target"]  # type: ignore[index]
    rng = np.random.default_rng(seed)
    map_params = fit_alignment("svd", hs, ht, params, rng)
    v_ref = transfer_direction(setup["v_source"], hs, ht, "svd", map_params)  # type: ignore[arg-type]
    ref_norm = float(np.linalg.norm(v_ref))
    v_rand = rng.normal(size=ht.shape[0])
    v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-8) * ref_norm
    unsafe_axis = setup["unsafe_axis_target"]  # type: ignore[index]
    unsafe_t = setup["unsafe_target"]  # type: ignore[index]
    scores = [
        unsafe_score_from_hidden(steer_hidden(h[np.newaxis, :], v_rand, alpha=alpha), unsafe_axis)
        for h in unsafe_t
    ]
    return attack_success_rate(np.array(scores))
