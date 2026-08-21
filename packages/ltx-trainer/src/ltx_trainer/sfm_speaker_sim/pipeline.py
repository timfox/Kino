"""Framework card, regression table, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sfm_speaker_sim.config import SfmSpeakerSimConfig
from ltx_trainer.sfm_speaker_sim.embedding import cosine_similarity_matrix
from ltx_trainer.sfm_speaker_sim.metrics import correspondence_bundle
from ltx_trainer.sfm_speaker_sim.models import model_taxonomy, representative_models
from ltx_trainer.sfm_speaker_sim.regression import table1_regression


def framework_card(cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    c = cfg or SfmSpeakerSimConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "human_vs_model_speaker_similarity",
        "models": f"{c.n_models} speech/audio foundation models",
        "datasets": list(c.datasets),
        "metrics": ["LCC", "SRCC", "Frobenius", "spectral"],
        "regression": "Table 1 model-configuration multiple regression",
        "headline": headline_results(c),
    }


def headline_results(cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    c = cfg or SfmSpeakerSimConfig()
    return {
        "n_models": c.n_models,
        "spectral_k": c.spectral_k,
        "wavlm_large_layer_max_lcc": c.wavlm_large_layer_max_lcc,
        "qwen3_tts_layer_max_lcc": c.qwen3_tts_layer_max_lcc,
        "reg_is_dec_lcc": c.reg_is_dec_lcc,
        "reg_is_ssl_lcc": c.reg_is_ssl_lcc,
        "reg_layer_max_lcc_r2": c.reg_layer_max_lcc_r2,
    }


def benchmarks_bundle(cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    c = cfg or SfmSpeakerSimConfig()
    return {
        "taxonomy": model_taxonomy(c),
        "table1_regression": table1_regression(c),
        "representative_layer_max": [
            {"model": p.name, "layer_max_lcc": p.layer_max_lcc, "layer_slope_lcc": p.layer_slope_lcc}
            for p in representative_models(c)
        ],
    }


def _synthetic_human_graph(n_speakers: int, rng: np.random.Generator) -> np.ndarray:
    latents = rng.standard_normal((n_speakers, 8))
    raw = latents @ latents.T
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    np.fill_diagonal(raw, 0.0)
    return raw


def _synthetic_model_graph(human: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    model = human + rng.standard_normal(human.shape) * noise
    model = np.clip(model, 0.0, 1.0)
    np.fill_diagonal(model, 0.0)
    return (model + model.T) / 2.0


def pipeline_demo(seed: int = 42, cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    """CPU stub: synthetic human/model graphs + correspondence metrics."""
    c = cfg or SfmSpeakerSimConfig()
    rng = np.random.default_rng(seed)
    n = 12
    human = _synthetic_human_graph(n, rng)
    low_noise = _synthetic_model_graph(human, 0.05, rng)
    high_noise = _synthetic_model_graph(human, 0.35, rng)

    good = correspondence_bundle(human, low_noise, spectral_k=c.spectral_k)
    poor = correspondence_bundle(human, high_noise, spectral_k=c.spectral_k)

    profiles = representative_models(c)
    wavlm = next(p for p in profiles if p.name == "wavlm-large")
    qwen = next(p for p in profiles if p.name == "qwen3-tts-12hz-0.6b-base")

    from ltx_trainer.sfm_speaker_sim.embedding import speaker_embedding

    embs = np.stack([speaker_embedding(rng.standard_normal((40, 16))) for _ in range(6)])
    sims = cosine_similarity_matrix(embs)

    reg = table1_regression(c)[0]
    return {
        "synthetic_good_lcc": round(good["lcc"], 4),
        "synthetic_poor_lcc": round(poor["lcc"], 4),
        "lcc_gain_low_vs_high_noise": round(good["lcc"] - poor["lcc"], 4),
        "wavlm_layer_max_lcc": wavlm.layer_max_lcc,
        "qwen3_tts_layer_max_lcc": qwen.layer_max_lcc,
        "wavlm_beats_qwen": wavlm.layer_max_lcc > qwen.layer_max_lcc,
        "reg_is_dec_lcc": reg["is_dec"],
        "reg_layer_max_lcc_r2": reg["r2"],
        "n_models": c.n_models,
        "embedding_sim_shape": list(sims.shape),
    }


def evaluation_demo(seed: int = 42, cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
