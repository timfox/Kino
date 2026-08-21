"""Table 1 — model-configuration multiple regression anchors."""

from __future__ import annotations

from typing import Any

from ltx_trainer.sfm_speaker_sim.config import SfmSpeakerSimConfig


def table1_regression(cfg: SfmSpeakerSimConfig | None = None) -> list[dict[str, Any]]:
    """Paper Table 1 coefficient summaries (speech models, dataset fixed effects)."""
    c = cfg or SfmSpeakerSimConfig()
    return [
        {
            "response": "layer_max",
            "metric": "LCC",
            "is_dec": c.reg_is_dec_lcc,
            "is_dec_p": c.reg_is_dec_lcc_p,
            "is_mlang": 0.00,
            "is_mlang_p": 0.992,
            "is_ssl": c.reg_is_ssl_lcc,
            "is_ssl_p": c.reg_is_ssl_lcc_p,
            "hours": -0.12,
            "hours_p": 0.117,
            "params": c.reg_params_lcc,
            "params_p": c.reg_params_lcc_p,
            "r2": c.reg_layer_max_lcc_r2,
        },
        {
            "response": "layer_max",
            "metric": "SRCC",
            "is_dec": -0.79,
            "is_dec_p": 0.001,
            "is_mlang": -0.03,
            "is_mlang_p": 0.687,
            "is_ssl": -0.15,
            "is_ssl_p": 0.015,
            "hours": -0.10,
            "hours_p": 0.182,
            "params": -0.15,
            "params_p": 0.010,
            "r2": 0.764,
        },
        {
            "response": "layer_max",
            "metric": "Neg. Frobenius",
            "is_dec": -0.22,
            "is_dec_p": 0.001,
            "is_mlang": -0.02,
            "is_mlang_p": 0.407,
            "is_ssl": -0.01,
            "is_ssl_p": 0.359,
            "hours": -0.01,
            "hours_p": 0.568,
            "params": 0.05,
            "params_p": 0.002,
            "r2": 0.983,
        },
        {
            "response": "layer_max",
            "metric": "Neg. spectral",
            "is_dec": 0.03,
            "is_dec_p": 0.529,
            "is_mlang": -0.16,
            "is_mlang_p": 0.008,
            "is_ssl": -0.15,
            "is_ssl_p": 0.003,
            "hours": 0.10,
            "hours_p": 0.096,
            "params": -0.03,
            "params_p": 0.550,
            "r2": 0.835,
        },
        {
            "response": "layer_slope",
            "metric": "LCC",
            "is_dec": c.reg_is_dec_slope_lcc,
            "is_dec_p": c.reg_is_dec_slope_lcc_p,
            "is_mlang": 0.10,
            "is_mlang_p": 0.425,
            "is_ssl": 0.06,
            "is_ssl_p": 0.610,
            "hours": -0.26,
            "hours_p": 0.054,
            "params": c.reg_params_slope_lcc,
            "params_p": c.reg_params_slope_lcc_p,
            "r2": c.reg_layer_slope_lcc_r2,
        },
    ]
