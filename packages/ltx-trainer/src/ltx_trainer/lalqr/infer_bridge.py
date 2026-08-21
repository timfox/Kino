"""LTX inference bridge: text-context LA-LQR proxy steering."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from ltx_trainer.lalqr.config import LalqrConfig
from ltx_trainer.lalqr.contrastive_prompts import CONTRASTIVE_SNIPPETS
from ltx_trainer.lalqr.lqr import la_lqr_text_control, solve_ltv_lqr
from ltx_trainer.lalqr.setpoints import latent_feature_direction, llfs_setpoint, tracking_error
from ltx_trainer.lalqr.steering import synthetic_local_dynamics, synthesize_contrastive_activations
from ltx_trainer.lalqr.subspace import (
    contrastive_rows,
    project_activation,
    randomized_svd_basis,
)

_LOG = logging.getLogger(__name__)

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Pornography": ("nude", "naked", "breast", "sexual", "porn", "explicit"),
    "Gore": ("blood", "gore", " severed", "wound", "corpse", "decapit"),
    "Public Figure": ("obama", "trump", "merkel", "pope", "celebrity", "famous"),
    "Copyright & Trademarks": ("nike", "apple iphone", "disney", "marvel", "coca-cola", "adidas"),
    "Sequential Action Risk": ("gun", "tiger enclosure", "loaded handgun", "precarious", "knife juggling"),
    "Violence": ("shooting", "gun", "stab", "massacre", "murder"),
    "Terrorism": ("bomb", "explosive", "terrorist", "countdown", "vest"),
    "Racism": ("racist", "hate symbol", "slur", "supremacist"),
    "Sexual": ("nude", "naked", "sexual", "porn"),
    "Animal Abuse": ("kick the dog", "animal abuse", "neglect", "chained dog", "cruelty"),
}


def infer_lalqr_enabled() -> bool:
    return os.environ.get("GOPEX_INFER_LALQR", "").strip().lower() in ("1", "true", "yes")


def infer_lalqr_model() -> str:
    return os.environ.get("GOPEX_LALQR_MODEL", "Wan2.1-T2V-14B LightX2V")


def infer_lalqr_category(prompt: str) -> str | None:
    explicit = os.environ.get("GOPEX_LALQR_CATEGORY", "").strip()
    if explicit:
        return explicit
    if os.environ.get("GOPEX_LALQR_AUTO_CATEGORY", "1").strip().lower() in ("0", "false", "no"):
        return os.environ.get("GOPEX_LALQR_DEFAULT_CATEGORY", "Pornography")
    text = prompt.lower()
    best: tuple[int, str | None] = (0, None)
    for cat, keys in _CATEGORY_KEYWORDS.items():
        score = sum(1 for k in keys if k in text)
        if score > best[0]:
            best = (score, cat)
    return best[1]


def _safety_lambda(weights_lambda: float) -> float:
    """Safety steering suppresses harmful contrastive direction (negative setpoint by default)."""
    override = os.environ.get("GOPEX_LALQR_LAMBDA", "").strip()
    if override:
        return float(override)
    return -abs(weights_lambda)


def _category_seed(category: str) -> int:
    return sum(ord(c) for c in category) % 10_000


def build_proxy_controller(
    cfg: LalqrConfig,
    *,
    category: str,
    model: str,
    embed_dim: int,
) -> dict[str, Any]:
    """Precompute basis + LQR gains using synthetic contrastive activations at embed_dim."""
    demo_cfg = LalqrConfig(
        demo_activation_dim=embed_dim,
        latent_rank=min(cfg.latent_rank, embed_dim, cfg.contrastive_pairs),
        contrastive_pairs=cfg.contrastive_pairs,
        text_dim=embed_dim,
        horizon_layers=cfg.horizon_layers,
    )
    seed = _category_seed(category) + cfg.svd_seed
    pos, neg = synthesize_contrastive_activations(demo_cfg, seed=seed)
    rows = contrastive_rows(pos, neg)
    basis = randomized_svd_basis(
        rows,
        rank=demo_cfg.latent_rank,
        oversampling=cfg.svd_oversampling,
        seed=seed,
    )
    ez, vz = latent_feature_direction(basis, rows)
    weights = cfg.lqr_for(model=model, category=category)
    lam = _safety_lambda(weights.lambda_setpoint)
    beta_star = llfs_setpoint(ez, lambda_setpoint=lam)
    d_lat = basis.shape[1]
    d_text = embed_dim
    a_list, b_list = synthetic_local_dynamics(
        horizon=cfg.horizon_layers,
        d_lat=d_lat,
        d_text=d_text,
        seed=seed + 3,
    )
    gains = solve_ltv_lqr(a_list, b_list, q=weights.q, r=weights.r_text, q_terminal=weights.q_terminal)
    return {
        "category": category,
        "basis": basis,
        "vz": vz,
        "beta_star": beta_star,
        "gain": gains[0],
        "rho": float(np.linalg.norm(basis.T @ rows.mean(axis=0)) ** 2 / (np.linalg.norm(rows.mean(axis=0)) ** 2 + 1e-12)),
        "lambda_effective": lam,
    }


def steer_text_context_numpy(
    context: np.ndarray,
    controller: dict[str, Any],
    *,
    pool: str = "mean",
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply LA-LQR correction to (T, D) or (D,) text / connector embeddings."""
    ctx = np.asarray(context, dtype=np.float64)
    if ctx.ndim == 1:
        pooled = ctx
        out = ctx.copy()
    else:
        pooled = ctx.mean(axis=0) if pool == "mean" else ctx[0]
        out = ctx.copy()

    basis = controller["basis"]
    vz = controller["vz"]
    z = project_activation(basis, pooled)
    alpha = tracking_error(z, vz, float(controller["beta_star"]))
    u = la_lqr_text_control(controller["gain"], vz, alpha)
    scale = float(os.environ.get("GOPEX_LALQR_STRENGTH", "1.0"))
    u = u * scale

    if ctx.ndim == 1:
        out = out + u
    else:
        out = out + u.reshape(1, -1)
    return out, {"alpha": alpha, "control_norm": float(np.linalg.norm(u)), "rho": float(controller["rho"])}


def steer_text_context_torch(context, *, prompt: str, cfg: LalqrConfig | None = None):
    """Torch wrapper for EmbeddingsProcessorOutput.video_encoding."""
    import torch

    cfg = cfg or LalqrConfig()
    category = infer_lalqr_category(prompt)
    if category is None:
        return context, {"skipped": True}

    if not isinstance(context, torch.Tensor):
        raise TypeError("context must be a torch.Tensor")

    embed_dim = int(context.shape[-1])
    ctrl = build_proxy_controller(cfg, category=category, model=infer_lalqr_model(), embed_dim=embed_dim)
    np_ctx = context.detach().float().cpu().numpy()
    steered, meta = steer_text_context_numpy(np_ctx, ctrl)
    delta = torch.from_numpy(steered - np_ctx).to(device=context.device, dtype=context.dtype)
    return context + delta, {"category": category, **meta}


def ltx_integration_notes(cfg: LalqrConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LalqrConfig()
    return {
        "use_case": "Inference-time T2V safety steering via reduced-order LA-LQR on text context",
        "hook_point": "PromptEncoder video_encoding (cross-attn context) before stage-1 denoise",
        "full_paper_path": "Per-layer DiT activation JVP + stored V_r bases — requires white-box capture",
        "proxy_path": "Contrastive subspace + Riccati gains on pooled connector embeddings (this stub)",
        "env": {
            "GOPEX_INFER_LALQR": "1 enables steering",
            "GOPEX_LALQR_CATEGORY": "explicit safety category",
            "GOPEX_LALQR_LAMBDA": "override LLFS λ (negative suppresses harmful direction)",
            "GOPEX_LALQR_STRENGTH": "scalar on control norm (default 1.0)",
            "GOPEX_LALQR_MODEL": "Wan vs Hunyuan LQR weights",
        },
        "categories": list(CONTRASTIVE_SNIPPETS.keys()),
        "latent_rank": cfg.latent_rank,
        "does_not": "Does not materialize 87M×87M Jacobians or per-layer video-token hooks in this repo stub",
    }


def apply_lalqr_to_prompt_contexts(
    v_context_p,
    v_context_n,
    *,
    prompt: str,
    negative_prompt: str,
):
    """Steer positive (and optionally negative) video contexts when enabled."""
    if not infer_lalqr_enabled():
        return v_context_p, v_context_n, {"enabled": False}

    cfg = LalqrConfig()
    v_p, meta_p = steer_text_context_torch(v_context_p, prompt=prompt, cfg=cfg)
    meta: dict[str, Any] = {"enabled": True, "positive": meta_p}
    v_n = v_context_n
    if os.environ.get("GOPEX_LALQR_STEER_NEGATIVE", "").strip().lower() in ("1", "true", "yes"):
        v_n, meta_n = steer_text_context_torch(v_context_n, prompt=negative_prompt, cfg=cfg)
        meta["negative"] = meta_n
    _LOG.info(
        "LA-LQR text steering: category=%s alpha=%.4f ||u||=%.4f",
        meta_p.get("category"),
        meta_p.get("alpha", 0.0),
        meta_p.get("control_norm", 0.0),
    )
    return v_p, v_n, meta
