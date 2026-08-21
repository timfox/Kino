"""FMelCodec framework card and paper tables (arXiv:2605.25669)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fmelcodec.cfm import cfm_loss, euler_step, linear_path, path_velocity, refine_mel_euler
from ltx_trainer.fmelcodec.coding import coding_stage_loss
from ltx_trainer.fmelcodec.config import FMelCodecConfig
from ltx_trainer.fmelcodec.layout import LIMITATIONS
from ltx_trainer.fmelcodec.oc_vq import bitrate_bps, quantize_nearest


def framework_card(cfg: FMelCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FMelCodecConfig()
    return {
        "name": "FMelCodec",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "github": cfg.github_repo,
        "idea": (
            "Three-stage CRR: mel coding (640× compression, OC-VQ) → CFM refinement "
            "(4-step ODE) → HiFi-GAN vocoding. 250 bps @ 16 kHz, 750 bps @ 48 kHz."
        ),
        "stages": [
            "ϕ_cod: ConvNeXt v2 encoder/decoder + single 1024-entry OC-VQ",
            "ϕ_ref: conditional flow matching with self-consistency training",
            "ϕ_voc: pretrained HiFi-GAN waveform reconstruction",
        ],
        "bitrates_bps": {"16kHz": cfg.bitrate_16k_bps, "48kHz": cfg.bitrate_48k_bps},
        "complexity_16k": {
            "params_m": cfg.params_m,
            "gflops": cfg.gflops_16k,
            "rtf": cfg.rtf_16k,
        },
        "defaults": cfg.__dict__,
    }


def table_i_equal_bitrate() -> list[dict[str, Any]]:
    """Table I excerpt — LibriTTS 250 bps / VCTK 750 bps."""
    return [
        {
            "method": "DAC",
            "libritts_visqol": 2.79,
            "libritts_dwer": 72.58,
            "libritts_nmos": 2.37,
            "vctk_visqol": 3.28,
            "vctk_dwer": 33.07,
        },
        {
            "method": "MDCTCodec",
            "libritts_visqol": 3.45,
            "libritts_dwer": 29.27,
            "libritts_nmos": 3.13,
            "vctk_visqol": 3.48,
            "vctk_dwer": 9.62,
        },
        {
            "method": "BigCodec",
            "libritts_visqol": 3.22,
            "libritts_dwer": 41.26,
            "libritts_nmos": 3.74,
            "vctk_visqol": 3.34,
            "vctk_dwer": 10.84,
        },
        {
            "method": "FocalCodec",
            "libritts_visqol": 3.12,
            "libritts_dwer": 4.97,
            "libritts_nmos": 3.65,
            "vctk_visqol": None,
            "vctk_dwer": None,
        },
        {
            "method": "FMelCodec",
            "libritts_visqol": 3.56,
            "libritts_dwer": 27.01,
            "libritts_nmos": 3.72,
            "vctk_visqol": 3.62,
            "vctk_dwer": 4.80,
        },
    ]


def table_ii_complexity() -> list[dict[str, Any]]:
    """Table II — RTF, GFLOPs, parameters on LibriTTS 16 kHz."""
    return [
        {"method": "DAC", "rtf": 0.096, "gflops": 32.22, "params_m": 73.96},
        {"method": "MDCTCodec", "rtf": 0.013, "gflops": 2.49, "params_m": 6.61},
        {"method": "BigCodec", "rtf": 0.052, "gflops": 28.03, "params_m": 158.31},
        {"method": "FocalCodec", "rtf": 0.026, "gflops": 8.84, "params_m": 143.30},
        {"method": "FMelCodec", "rtf": 0.022, "gflops": 18.47, "params_m": 27.17},
    ]


def table_iii_public_checkpoints() -> list[dict[str, Any]]:
    """Table III — FMelCodec 250 bps vs public ultra-low-bitrate checkpoints."""
    return [
        {"method": "FocalCodec†", "bitrate_bps": 330, "visqol": 3.49, "dwer": 3.21, "nmos": 3.86},
        {"method": "SemantiCodec†", "bitrate_bps": 310, "visqol": 3.32, "dwer": 44.82, "nmos": 3.21},
        {"method": "FMelCodec", "bitrate_bps": 250, "visqol": 3.56, "dwer": 27.01, "nmos": 3.79},
    ]


def table_iv_stage_complexity() -> dict[str, Any]:
    """Table IV — per-stage GFLOPs and parameter share."""
    return {
        "phi_cod": {"gflops": 0.60, "gflops_pct": 3.25, "params_m": 6.29, "params_pct": 23.15},
        "phi_ref": {"gflops": 1.48, "gflops_pct": 8.02, "params_m": 7.84, "params_pct": 28.86},
        "phi_voc": {"gflops": 16.38, "gflops_pct": 88.73, "params_m": 13.04, "params_pct": 47.99},
    }


def headline_results() -> dict[str, Any]:
    return {
        "bitrate_16k_bps": 250,
        "bitrate_48k_bps": 750,
        "compression_factor": "640× waveform-level via mel hop + r=4",
        "libritts_nmos_smos": "3.72 / 3.51",
        "bitrate_savings_vs_baselines": "approximately 250–750 bps (ABX, Sec. IV-G)",
    }


def pipeline_demo(cfg: FMelCodecConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or FMelCodecConfig()
    rng = np.random.default_rng(seed)
    mel = rng.standard_normal((32, cfg.mel_bins_16k))
    coarse = mel + 0.15 * rng.standard_normal(mel.shape)
    z = rng.standard_normal(cfg.latent_dim)
    codebook = rng.standard_normal((cfg.codebook_size, cfg.latent_dim))
    idx, z_q = quantize_nearest(z, codebook)

    m0 = rng.standard_normal(mel.shape)
    target = mel
    t = 0.3
    mt = linear_path(m0, target, t)
    v_true = path_velocity(target, m0)
    v_pred = v_true + 0.05 * rng.standard_normal(v_true.shape)
    l_cod = coding_stage_loss(mel, coarse, z, z_q, lambda_mel_rec=cfg.lambda_mel_rec, lambda_vq=cfg.lambda_vq)
    l_cfm = cfm_loss(v_pred, target, m0)

    def velocity_fn(state, time, _cond):
        return path_velocity(target, m0)

    refined = refine_mel_euler(m0, velocity_fn, steps=cfg.cfm_ode_steps)

    return {
        "bitrate_bps_16k": bitrate_bps(
            sample_rate=16000,
            temporal_downsample=cfg.temporal_downsample,
            frame_shift=cfg.frame_shift,
            codebook_size=cfg.codebook_size,
        ),
        "token_index": idx,
        "coding_loss": l_cod,
        "cfm_loss": l_cfm,
        "refined_mel_shape": list(refined.shape),
        "euler_steps": cfg.cfm_ode_steps,
    }


def evaluation_demo(cfg: FMelCodecConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["headline"] = headline_results()
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_equal_bitrate": table_i_equal_bitrate(),
        "table_ii_complexity": table_ii_complexity(),
        "table_iii_public_checkpoints": table_iii_public_checkpoints(),
        "table_iv_stage_complexity": table_iv_stage_complexity(),
        "headline_results": headline_results(),
        "crr_framework": ["phi_cod", "phi_ref", "phi_voc"],
    }
