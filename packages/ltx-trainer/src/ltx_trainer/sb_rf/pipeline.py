"""Framework card, Table 1/2 benchmarks, CPU demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sb_rf.bridge import bb_sample_xt, rf_linear_path, sb_sample_xt
from ltx_trainer.sb_rf.config import SbRfConfig
from ltx_trainer.sb_rf.flow import one_step_enhance, velocity_matching_loss, velocity_target
from ltx_trainer.sb_rf.loss import amplitude_transform, composite_loss


def framework_card(cfg: SbRfConfig | None = None) -> dict[str, Any]:
    c = cfg or SbRfConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "one_step_speech_enhancement",
        "tracks": ["VB-DMD (Track A)", "low-SNR robustness (Track B)"],
        "nfe": c.nfe_default,
        "backbone": f"NCSN++ {c.backbone_params_m}M params, {c.backbone_gmacs:g} GMACs",
        "loss": f"Lv + {c.lambda_mel:g} Lmel + {c.lambda_pesq:g} Lpesq",
        "headline": headline_results(c),
    }


def headline_results(cfg: SbRfConfig | None = None) -> dict[str, Any]:
    c = cfg or SbRfConfig()
    return {
        "track_a_pesq": c.sb_rf_track_a_pesq,
        "track_a_si_sdr": c.sb_rf_track_a_si_sdr,
        "track_b_pesq": c.sb_rf_track_b_pesq,
        "track_b_estoi": c.sb_rf_track_b_estoi,
        "nfe": c.nfe_default,
        "gain_vs_cose_track_a": c.track_a_pesq_gain_vs_cose,
        "gain_vs_bb_rf_track_b": c.track_b_pesq_gain_vs_bb_rf,
    }


def table1_vbdmd(cfg: SbRfConfig | None = None) -> list[dict[str, Any]]:
    """Table 1 — VB-DMD Track A."""
    c = cfg or SbRfConfig()
    return [
        {"method": "noisy", "nfe": 1, "pesq": c.noisy_track_a_pesq, "estoi": 0.79, "si_sdr": 8.4},
        {"method": "SGMSE+", "nfe": 15, "pesq": 2.80, "estoi": 0.86, "si_sdr": 17.2},
        {"method": "BBED", "nfe": 60, "pesq": 3.09, "estoi": 0.88, "si_sdr": 18.8},
        {"method": "SB-VE", "nfe": 50, "pesq": 2.91, "estoi": 0.88, "si_sdr": 19.4},
        {"method": "CFM", "nfe": 5, "pesq": c.cfm_track_a_pesq, "estoi": 0.88, "si_sdr": 19.0},
        {"method": "COSE", "nfe": 1, "pesq": c.cose_track_a_pesq, "estoi": 0.87, "si_sdr": 19.3},
        {"method": "LARF", "nfe": 1, "pesq": 2.97, "estoi": 0.87, "si_sdr": 19.2},
        {"method": "BB-RF", "nfe": 1, "pesq": c.bb_rf_track_a_pesq, "estoi": 0.87, "si_sdr": 18.9},
        {
            "method": "SB-RF",
            "nfe": 1,
            "pesq": c.sb_rf_track_a_pesq,
            "estoi": c.sb_rf_track_a_estoi,
            "si_sdr": c.sb_rf_track_a_si_sdr,
            "si_sir": c.sb_rf_track_a_si_sir,
            "si_sar": c.sb_rf_track_a_si_sar,
        },
    ]


def table2_low_snr(cfg: SbRfConfig | None = None) -> list[dict[str, Any]]:
    """Table 2 — Track B low-SNR test set."""
    c = cfg or SbRfConfig()
    return [
        {"method": "noisy", "nfe": 1, "pesq": 1.12, "estoi": 0.36, "si_sdr": -5.4, "dnsmos": 2.42},
        {"method": "MP-SENet", "nfe": 1, "pesq": c.mp_senet_track_b_pesq, "estoi": 0.66, "si_sdr": c.mp_senet_track_b_si_sdr, "dnsmos": 3.29},
        {"method": "BBED", "nfe": 60, "pesq": 1.83, "estoi": 0.62, "si_sdr": 7.8, "dnsmos": 3.49},
        {"method": "SB-VE", "nfe": 50, "pesq": 2.07, "estoi": 0.66, "si_sdr": 9.1, "dnsmos": 3.42},
        {"method": "BB-RF", "nfe": 1, "pesq": c.bb_rf_track_b_pesq, "estoi": 0.66, "si_sdr": 9.1, "dnsmos": 3.36},
        {
            "method": "SB-RF",
            "nfe": 1,
            "pesq": c.sb_rf_track_b_pesq,
            "estoi": c.sb_rf_track_b_estoi,
            "si_sdr": c.sb_rf_track_b_si_sdr,
            "dnsmos": c.sb_rf_track_b_dnsmos,
        },
        {"method": "SB-RF", "nfe": 5, "pesq": 2.46, "estoi": 0.70, "si_sdr": 10.7, "dnsmos": 3.42},
        {"method": "SB-RF", "nfe": 10, "pesq": 2.44, "estoi": 0.70, "si_sdr": 10.7, "dnsmos": 3.43},
    ]


def benchmarks_bundle(cfg: SbRfConfig | None = None) -> dict[str, Any]:
    c = cfg or SbRfConfig()
    return {
        "track_a_vbdmd": table1_vbdmd(c),
        "track_b_low_snr": table2_low_snr(c),
        "training": {
            "track_a_utterances": c.track_a_train_utterances,
            "track_b_hours": c.track_b_train_hours,
            "track_b_test_utterances": c.track_b_test_utterances,
        },
    }


def pipeline_demo(seed: int = 42, cfg: SbRfConfig | None = None) -> dict[str, Any]:
    c = cfg or SbRfConfig()
    rng = np.random.default_rng(seed)
    shape = (c.freq_bins, c.time_frames)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    y = x + 0.3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    x = amplitude_transform(x, c.amplitude_alpha, c.amplitude_beta)
    y = amplitude_transform(y, c.amplitude_alpha, c.amplitude_beta)

    t = 0.5
    xt_sb = sb_sample_xt(x, y, t, rng)
    xt_bb = bb_sample_xt(x, y, t, rng)
    xt_rf = rf_linear_path(x, y, t)

    v_target = velocity_target(x, y)
    v_pred = v_target + 0.05 * rng.standard_normal(v_target.shape)
    losses = composite_loss(
        v_pred, x, y, xt_rf, t,
        epsilon=c.epsilon,
        lambda_mel=c.lambda_mel,
        lambda_pesq=c.lambda_pesq,
    )

    def ideal_v(xt, t_val, y_obs):
        return velocity_target(x, y_obs)

    x_hat = one_step_enhance(y, ideal_v, t_start=c.t_max, epsilon=c.epsilon)
    sb_rf_a = next(r for r in table1_vbdmd(c) if r["method"] == "SB-RF")
    bb_rf_a = next(r for r in table1_vbdmd(c) if r["method"] == "BB-RF")

    return {
        "velocity_loss": round(losses["velocity"], 6),
        "composite_loss": round(losses["total"], 4),
        "one_step_mse_vs_clean": round(float(np.mean(np.abs(x_hat - x) ** 2)), 6),
        "sb_rf_track_a_pesq": sb_rf_a["pesq"],
        "bb_rf_track_a_pesq": bb_rf_a["pesq"],
        "sb_rf_beats_bb_rf_track_a": sb_rf_a["pesq"] > bb_rf_a["pesq"],
        "sb_rf_track_b_pesq": c.sb_rf_track_b_pesq,
        "nfe": c.nfe_default,
        "xt_sb_shape": list(xt_sb.shape),
        "xt_bb_shape": list(xt_bb.shape),
    }


def evaluation_demo(seed: int = 42, cfg: SbRfConfig | None = None) -> dict[str, Any]:
    return pipeline_demo(seed=seed, cfg=cfg)
