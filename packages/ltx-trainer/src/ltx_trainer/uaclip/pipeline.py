"""Utility-Aware CLIP pipeline and evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.uaclip.baselines import (
    TABLE2_AMAZON_EDITING,
    TABLE5_AIRBNB_EDITING,
    TABLE6_HUMAN_AMAZON,
)
from ltx_trainer.uaclip.config import UAClipConfig
from ltx_trainer.uaclip.demand import Platform, amazon_optimal_attrs, visual_utility
from ltx_trainer.uaclip.generator import CandidateImage, select_best
from ltx_trainer.uaclip.infonce import bidirectional_infonce, build_score_matrix
from ltx_trainer.uaclip.occlusion import patch_occlusion_sensitivity
from ltx_trainer.uaclip.similarity import l2_normalize
from ltx_trainer.uaclip.theory import utility_aware_infonce_bound
from ltx_trainer.uaclip.visual_attrs import VisualAttributes


def framework_card(cfg: UAClipConfig | None = None) -> dict[str, Any]:
    cfg = cfg or UAClipConfig()
    return {
        "name": "Utility-Aware CLIP",
        "title": "Utility-Aware Multimodal Contrastive Learning for Product Image Generation",
        "arxiv": cfg.arxiv,
        "authors": "Feng & Xie (City University of Hong Kong)",
        "components": [
            "utility_aware_infonce (Eq. 4)",
            "utility_aware_clip_score (Eq. 3)",
            "demand_models_amazon_airbnb",
            "utility_aware_generator_ranking",
            "occlusion_interpretability",
        ],
        "config": cfg.__dict__,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "problem": (
            "CLIP optimizes semantic alignment, not marketplace demand; pretty images ≠ images that sell."
        ),
        "solution": "Regularize contrastive similarity with demand-driven visual utility h_v(v).",
        "platforms": {
            "amazon": "colorfulness, brightness, symmetry, aesthetic (inverted-U on log sales rank)",
            "airbnb": "uniqueness, aesthetic (inverted-U on log occupancy)",
        },
        "baselines": ["Stable Diffusion", "GPT-Image", "Flux"],
        "limitations_stub": "Numpy reference — no Flux/CLIP fine-tuning or real image generation.",
    }


def benchmarks_bundle() -> dict[str, Any]:
    from ltx_trainer.uaclip.baselines import (
        TABLE1_AMAZON_DEMAND,
        TABLE3_AIRBNB_DEMAND,
        TABLE4_AIRBNB_GENERATION,
        TABLE7_REGRESSION_AMAZON,
        TABLE8_AIRBNB_RATINGS,
        TABLE9_REGRESSION_AIRBNB,
    )

    return {
        "table1_amazon_demand": TABLE1_AMAZON_DEMAND,
        "table2_amazon_editing": TABLE2_AMAZON_EDITING,
        "table3_airbnb_demand": TABLE3_AIRBNB_DEMAND,
        "table4_airbnb_generation": TABLE4_AIRBNB_GENERATION,
        "table5_airbnb_editing": TABLE5_AIRBNB_EDITING,
        "table6_human_amazon": TABLE6_HUMAN_AMAZON,
        "table7_regression_amazon": TABLE7_REGRESSION_AMAZON,
        "table8_airbnb_ratings": TABLE8_AIRBNB_RATINGS,
        "table9_regression_airbnb": TABLE9_REGRESSION_AIRBNB,
    }


def _synthetic_batch(
    n: int,
    d: int,
    platform: Platform,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_embs = l2_normalize(rng.standard_normal((n, d)))
    image_embs = l2_normalize(rng.standard_normal((n, d)))
    utilities = np.zeros(n)
    for i in range(n):
        attrs = VisualAttributes.random(rng, platform)
        utilities[i] = visual_utility(attrs.to_dict(platform), platform, eta=0.5)
    return image_embs, text_embs, utilities


def evaluation_demo(
    *,
    seed: int = 0,
    platform: Platform = "amazon",
    include_train: bool = True,
) -> dict[str, Any]:
    cfg = UAClipConfig(embed_dim=64, temperature=1.0, eta_utility=0.5)
    rng = np.random.default_rng(seed)
    n = 16
    d = cfg.embed_dim
    image_embs, text_embs, utilities = _synthetic_batch(n, d, platform, rng)

    clip_scores = build_score_matrix(
        image_embs,
        text_embs,
        utilities,
        alpha_visual=cfg.alpha_visual,
        beta_semantic=cfg.beta_semantic,
        utility_aware=False,
    )
    ua_scores = build_score_matrix(
        image_embs,
        text_embs,
        utilities,
        alpha_visual=cfg.alpha_visual,
        beta_semantic=cfg.beta_semantic,
        utility_aware=True,
    )

    clip_loss = bidirectional_infonce(clip_scores, temperature=cfg.temperature, utility_aware=False)
    ua_loss = bidirectional_infonce(ua_scores, temperature=cfg.temperature, utility_aware=True)
    bound = utility_aware_infonce_bound(
        ua_loss["loss"],
        n,
        utilities,
        alpha_visual=cfg.alpha_visual,
    )

    ref_emb = image_embs[0]
    text_emb = text_embs[0]
    candidates = []
    for j in range(5):
        attrs = VisualAttributes.random(rng, platform)
        candidates.append(
            CandidateImage(
                candidate_id=f"cand-{j}",
                image_emb=l2_normalize(rng.standard_normal(d)),
                attrs=attrs,
            )
        )
    best = select_best(candidates, text_emb, ref_emb, platform, eta=cfg.eta_utility)

    attrs0 = VisualAttributes.from_vector(
        np.array(list(amazon_optimal_attrs().values()) if platform == "amazon" else [0.5, 0.6]),
        platform,
    )
    occ_clip = patch_occlusion_sensitivity(
        image_embs[0], text_emb, attrs0, platform, utility_aware=False
    )
    occ_ua = patch_occlusion_sensitivity(
        image_embs[0], text_emb, attrs0, platform, utility_aware=True
    )

    paper_row = next(
        r for r in (TABLE2_AMAZON_EDITING if platform == "amazon" else TABLE5_AIRBNB_EDITING)
        if r["model"] == "Utility-Aware Generator"
    )

    out: dict[str, Any] = {
        "platform": platform,
        "clip_infonce_loss": clip_loss["loss"],
        "utility_aware_infonce_loss": ua_loss["loss"],
        "theorem_4_1": bound,
        "best_candidate": best.to_dict(),
        "occlusion": {
            "clip_mean_delta": occ_clip.mean_delta,
            "uclip_mean_delta": occ_ua.mean_delta,
            "uclip_more_spread": float(occ_ua.delta_map.std()) > float(occ_clip.delta_map.std()),
        },
        "paper_uag_demand": paper_row["demand"],
        "human_selection_amazon": TABLE6_HUMAN_AMAZON[-1]["selection_count"],
    }
    if include_train:
        from ltx_trainer.uaclip.training import demo_train

        out["train_step"] = demo_train(seed=seed, platform=platform)
    from ltx_trainer.uaclip.compare import demand_at_optimal, score_candidate_batch

    out["demand_at_optimal"] = demand_at_optimal(platform)
    out["candidate_batch"] = score_candidate_batch(platform, seed=seed + 1)
    return out
