"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.edr_zs_ser.config import EdrZsSerConfig
from ltx_trainer.edr_zs_ser.losses import (
    emotion_ce_loss,
    speaker_adversarial_loss,
    supervised_contrastive_loss,
    total_loss,
)


def framework_card(cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    c = cfg or EdrZsSerConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "zero_shot_cross_lingual_ser",
        "feature_extractor": c.feature_extractor,
        "emotion_classes": list(c.emotion_classes),
        "components": [
            "Language-matched wav2vec2 + mean pooling",
            "Language-aware supervised contrastive learning (λ>1 cross-lingual pairs)",
            "GRL speaker adversarial classifier",
            "Linear emotion classifier (4 classes)",
        ],
        "loss": "L = LCE + α LSupCLR + β LSpkAdv",
        "headline": headline_results(c),
    }


def headline_results(cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    c = cfg or EdrZsSerConfig()
    return {
        "proposed_avg_uar": c.proposed_avg_uar,
        "proposed_avg_f1": c.proposed_avg_f1,
        "baseline2_avg_uar": c.baseline2_avg_uar,
        "uar_gain_vs_baseline2": round(c.proposed_avg_uar - c.baseline2_avg_uar, 2),
        "f1_gain_vs_baseline2": round(c.proposed_avg_f1 - c.baseline2_avg_f1, 2),
        "en_de_proposed_uar": c.en_de_proposed_uar,
        "upper_bound_avg_uar": c.upper_bound_avg_uar,
    }


def table1_crosslingual_settings() -> list[dict[str, Any]]:
    """Table 1 — nine zero-shot cross-lingual task configurations."""
    return [
        {"task": "EN→DE", "source": "EN", "non_target": "CN, UR, FR", "target": "DE", "samples": 20165, "speakers": 288},
        {"task": "CN→DE", "source": "CN", "non_target": "EN, UR, FR", "target": "DE", "samples": 20165, "speakers": 288},
        {"task": "FR→DE", "source": "FR", "non_target": "EN, CN, UR", "target": "DE", "samples": 20165, "speakers": 288},
        {"task": "EN→FR", "source": "EN", "non_target": "CN, DE, UR", "target": "FR", "samples": 20011, "speakers": 286},
        {"task": "CN→FR", "source": "CN", "non_target": "EN, DE, UR", "target": "FR", "samples": 20011, "speakers": 286},
        {"task": "DE→FR", "source": "DE", "non_target": "EN, CN, UR", "target": "FR", "samples": 20011, "speakers": 286},
        {"task": "EN→CN", "source": "EN", "non_target": "DE, UR, FR", "target": "CN", "samples": 9231, "speakers": 288},
        {"task": "DE→CN", "source": "DE", "non_target": "EN, UR, FR", "target": "CN", "samples": 9231, "speakers": 288},
        {"task": "FR→CN", "source": "FR", "non_target": "EN, DE, UR", "target": "CN", "samples": 9231, "speakers": 288},
    ]


def table2_results(cfg: EdrZsSerConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — UAR/F1 (%) for all nine settings (selected + average row)."""
    c = cfg or EdrZsSerConfig()
    rows = [
        {"task": "EN→DE", "b1_uar": 52.23, "b2_uar": 88.19, "prop_uar": 94.64, "prop_f1": 94.36, "upper_uar": 97.22},
        {"task": "CN→DE", "b1_uar": 85.42, "b2_uar": 88.54, "prop_uar": 94.44, "prop_f1": 95.21, "upper_uar": 97.22},
        {"task": "FR→DE", "b1_uar": 73.91, "b2_uar": 80.95, "prop_uar": 88.89, "prop_f1": 89.73, "upper_uar": 95.44},
        {"task": "EN→FR", "b1_uar": 45.83, "b2_uar": 70.83, "prop_uar": 77.08, "prop_f1": 75.35, "upper_uar": 87.50},
        {"task": "CN→FR", "b1_uar": 72.92, "b2_uar": 79.17, "prop_uar": 85.42, "prop_f1": 86.86, "upper_uar": 89.58},
        {"task": "DE→FR", "b1_uar": 47.92, "b2_uar": 68.75, "prop_uar": 75.00, "prop_f1": 72.84, "upper_uar": 81.25},
        {"task": "EN→CN", "b1_uar": 58.36, "b2_uar": 71.14, "prop_uar": 78.07, "prop_f1": 77.92, "upper_uar": 97.07},
        {"task": "DE→CN", "b1_uar": 48.79, "b2_uar": 53.64, "prop_uar": 73.86, "prop_f1": 73.48, "upper_uar": 91.79},
        {"task": "FR→CN", "b1_uar": 50.07, "b2_uar": 57.64, "prop_uar": 72.93, "prop_f1": 71.87, "upper_uar": 90.21},
    ]
    rows.append(
        {
            "task": "Avg.",
            "b1_uar": c.baseline1_avg_uar,
            "b2_uar": c.baseline2_avg_uar,
            "prop_uar": c.proposed_avg_uar,
            "prop_f1": c.proposed_avg_f1,
            "wo_supclr_uar": c.proposed_wo_supclr_avg_uar,
            "wo_spkadv_uar": c.proposed_wo_spkadv_avg_uar,
            "upper_uar": c.upper_bound_avg_uar,
            "average": True,
        }
    )
    return rows


def benchmarks_bundle(cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    c = cfg or EdrZsSerConfig()
    return {
        "table1_crosslingual_settings": table1_crosslingual_settings(),
        "table2_results": table2_results(c),
        "hyperparameters": {
            "lambda": c.lambda_crossling,
            "alpha": c.alpha_supclr,
            "beta": c.beta_spk_adv,
            "n_lang": c.n_lang,
            "n_cls": c.n_cls,
            "n_sam": c.n_sam,
        },
    }


def evaluation_demo(seed: int = 42, cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or EdrZsSerConfig()
    batch = c.n_lang * c.n_cls * c.n_sam

    embeddings = rng.normal(0, 1, (batch, 32))
    emotions = np.tile(np.arange(c.n_cls), c.n_lang * c.n_sam)
    languages = np.repeat(np.array(["EN", "CN", "DE"]), c.n_cls * c.n_sam)

    supclr = supervised_contrastive_loss(
        embeddings,
        emotions,
        languages,
        temperature=c.contrastive_temperature,
        lam=c.lambda_crossling,
    )

    spk_logits = rng.normal(0, 0.1, (batch, 8))
    speakers = rng.integers(0, 8, batch)
    spk_loss = speaker_adversarial_loss(spk_logits, speakers)

    emo_logits = rng.normal(0, 0.1, (batch, len(c.emotion_classes)))
    emo_targets = emotions % len(c.emotion_classes)
    ce = emotion_ce_loss(emo_logits, emo_targets)

    loss = total_loss(ce, supclr, spk_loss, alpha=c.alpha_supclr, beta=c.beta_spk_adv)

    avg = next(r for r in table2_results(c) if r.get("average"))
    en_de = next(r for r in table2_results(c) if r["task"] == "EN→DE")

    return {
        "batch_size": batch,
        "supclr_loss": supclr,
        "spk_adv_loss": spk_loss,
        "ce_loss": ce,
        "total_loss": loss,
        "proposed_beats_baseline2_avg": avg["prop_uar"] > avg["b2_uar"],
        "uar_gain_vs_baseline2": round(c.proposed_avg_uar - c.baseline2_avg_uar, 2),
        "supclr_ablation_drop_uar": c.ablation_supclr_uar_drop,
        "spkadv_ablation_drop_uar": c.ablation_spkadv_uar_drop,
        "en_de_proposed_uar": en_de["prop_uar"],
    }


def pipeline_demo(seed: int = 42, cfg: EdrZsSerConfig | None = None) -> dict[str, Any]:
    c = cfg or EdrZsSerConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
