"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sfm_speaker_sim.config import SfmSpeakerSimConfig
from ltx_trainer.sfm_speaker_sim.embedding import normalize_scores
from ltx_trainer.sfm_speaker_sim.metrics import correspondence_bundle, frobenius_distance, pearson_lcc
from ltx_trainer.sfm_speaker_sim.pipeline import benchmarks_bundle, evaluation_demo
from ltx_trainer.sfm_speaker_sim.regression import table1_regression


def evaluation_smoke(cfg: SfmSpeakerSimConfig | None = None) -> dict[str, Any]:
    c = cfg or SfmSpeakerSimConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    reg = table1_regression(c)[0]

    assert c.n_models == 43
    assert c.spectral_k == 10
    assert c.wavlm_large_layer_max_lcc == 0.38
    assert c.qwen3_tts_layer_max_lcc == 0.12
    assert c.reg_is_dec_lcc == -0.77
    assert c.reg_layer_max_lcc_r2 == 0.747
    assert demo["wavlm_beats_qwen"] is True
    assert demo["lcc_gain_low_vs_high_noise"] > 0
    assert reg["is_dec"] < 0
    assert reg["is_ssl"] < 0
    assert len(benchmarks_bundle(c)["table1_regression"]) == 5

    raw = np.array([-3.0, 0.0, 3.0])
    norm = normalize_scores(raw, c.perceptual_score_min, c.perceptual_score_max)
    assert np.allclose(norm, [0.0, 0.5, 1.0])

    human = np.array([[0, 1, 0.5], [1, 0, 0.6], [0.5, 0.6, 0]], dtype=np.float64)
    model = np.array([[0, 0.95, 0.48], [0.95, 0, 0.58], [0.48, 0.58, 0]], dtype=np.float64)
    assert pearson_lcc(human, model) > 0.99
    assert frobenius_distance(human, model) < 0.2
    corr = correspondence_bundle(human, model, spectral_k=2)
    assert corr["lcc"] > 0.9

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "n_models": c.n_models,
        "wavlm_large_layer_max_lcc": c.wavlm_large_layer_max_lcc,
        "qwen3_tts_layer_max_lcc": c.qwen3_tts_layer_max_lcc,
        "reg_is_dec_lcc": c.reg_is_dec_lcc,
        "reg_layer_max_lcc_r2": c.reg_layer_max_lcc_r2,
    }
