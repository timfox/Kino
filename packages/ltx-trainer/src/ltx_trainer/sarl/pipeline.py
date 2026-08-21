"""Framework card, encoder table, probing demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sarl.config import SarlConfig
from ltx_trainer.sarl.probing import aggregate_group, baseline_normalize, linear_probe_predict
from ltx_trainer.sarl.sensitivity import batch_sensitivity, expected_random_similarity
from ltx_trainer.sarl.tasks import ENCODERS, ROOM_TASKS, SOURCE_TASKS


def framework_card(cfg: SarlConfig | None = None) -> dict[str, Any]:
    c = cfg or SarlConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "spatial_audio_representation_probing",
        "probe_tasks": len(SOURCE_TASKS) + len(ROOM_TASKS),
        "encoders": c.num_encoders,
        "headline": headline_results(c),
    }


def headline_results(cfg: SarlConfig | None = None) -> dict[str, Any]:
    c = cfg or SarlConfig()
    return {
        "strongest_encoder": "GRAM-F",
        "foa_beats_mono_localization": c.gram_f_localization > c.a_mae_localization,
        "source_exceeds_room": c.gram_f_localization > c.gram_f_room,
        "source_sensitivity_exceeds_room": c.gram_f_source_sensitivity > c.gram_f_room_sensitivity,
    }


def table1_encoders() -> list[dict[str, str]]:
    return [dict(row) for row in ENCODERS]


def fig2_aggregates(cfg: SarlConfig | None = None) -> list[dict[str, Any]]:
    """Fig. 2 style semantic / localization / room aggregates."""
    c = cfg or SarlConfig()
    return [
        {
            "model": "GRAM-F",
            "semantic": c.gram_f_semantic,
            "localization": c.gram_f_localization,
            "room": c.gram_f_room,
        },
        {
            "model": "A-MAE",
            "semantic": c.a_mae_semantic,
            "localization": c.a_mae_localization,
            "room": c.a_mae_room,
        },
        {
            "model": "SELD-S",
            "semantic": 0.79,
            "localization": c.seld_s_localization,
            "room": c.seld_s_room,
        },
        {
            "model": "EnCodec",
            "semantic": 0.55,
            "localization": c.encodec_localization,
            "room": c.encodec_room,
        },
        {
            "model": "EINv2",
            "semantic": 0.84,
            "localization": 0.76,
            "room": 0.44,
        },
    ]


def fig3_sensitivity(cfg: SarlConfig | None = None) -> list[dict[str, Any]]:
    c = cfg or SarlConfig()
    return [
        {
            "model": "GRAM-F",
            "source": c.gram_f_source_sensitivity,
            "room": c.gram_f_room_sensitivity,
        },
        {
            "model": "EINv2",
            "source": 0.35,
            "room": 0.19,
        },
        {
            "model": "SFD",
            "source": c.sfd_source_sensitivity,
            "room": c.sfd_room_sensitivity,
        },
        {
            "model": "BANC",
            "source": 0.68,
            "room": 0.52,
        },
    ]


def benchmarks_bundle(cfg: SarlConfig | None = None) -> dict[str, Any]:
    c = cfg or SarlConfig()
    return {
        "table1_encoders": table1_encoders(),
        "fig2_aggregates": fig2_aggregates(c),
        "fig3_sensitivity": fig3_sensitivity(c),
        "protocol": {
            "probe_lr": c.probe_lr,
            "probe_epochs": c.probe_epochs,
            "seeds": c.probe_seeds,
            "normalization": "phi(x,b)=(x-b)/(1-b)",
        },
    }


def _format_mean_localization(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for row in rows:
        fmt = next(m["input"] for m in ENCODERS if m["name"] == row["model"])
        out.setdefault(fmt, []).append(float(row["localization"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def pipeline_demo(seed: int = 42, cfg: SarlConfig | None = None) -> dict[str, Any]:
    c = cfg or SarlConfig()
    rng = np.random.default_rng(seed)

    embeddings = rng.standard_normal((128, 32))
    labels = rng.integers(0, c.event_classes, size=128)
    macro_f1 = linear_probe_predict(embeddings, labels, seed=seed)

    refs = rng.standard_normal((64, 16))
    source_perturbed = refs + rng.normal(0, 0.35, refs.shape)
    room_perturbed = refs + rng.normal(0, 0.12, refs.shape)
    mu = expected_random_similarity(refs, seed=seed)
    source_delta = batch_sensitivity(refs, source_perturbed, mu=mu)
    room_delta = batch_sensitivity(refs, room_perturbed, mu=mu)

    fig2 = fig2_aggregates(c)
    gram_f = next(r for r in fig2 if r["model"] == "GRAM-F")
    format_means = _format_mean_localization(fig2)

    azimuth_score = 0.74
    azimuth_baseline = baseline_normalize(azimuth_score, 0.0)
    class_score = 0.88
    class_baseline = baseline_normalize(class_score, 1.0 / c.event_classes)

    return {
        "linear_probe_macro_f1": round(macro_f1, 4),
        "source_sensitivity": round(source_delta, 4),
        "room_sensitivity": round(room_delta, 4),
        "source_more_sensitive": source_delta > room_delta,
        "gram_f_source_gt_room": gram_f["localization"] > gram_f["room"],
        "all_models_source_gt_room": all(r["localization"] > r["room"] for r in fig2),
        "foa_mean_localization": round(format_means.get("foa", 0.0), 4),
        "mono_mean_localization": round(format_means.get("mono", 0.0), 4),
        "foa_beats_mono": format_means.get("foa", 0.0) > format_means.get("mono", 0.0),
        "azimuth_phi": round(azimuth_baseline, 4),
        "class_phi": round(class_baseline, 4),
        "num_encoders": c.num_encoders,
        "num_tasks": len(SOURCE_TASKS) + len(ROOM_TASKS),
    }


def evaluation_demo(seed: int = 42, cfg: SarlConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
