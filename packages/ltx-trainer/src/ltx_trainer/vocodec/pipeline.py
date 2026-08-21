"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.vocodec.config import VoCodecConfig
from ltx_trainer.vocodec.quantizer import (
    bitrate_kbps_simplified,
    detect_voicing_flags,
    mask_based_quantize_batch,
    quantization_token_layout,
)


def framework_card(cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or VoCodecConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "streamable_neural_speech_codec",
        "components": [
            "Causal MDCT encoder-decoder (StreamCodec backbone)",
            "FFT energy voicing detector (Eq. 2–3)",
            "RSVQ for voiced / SQ for unvoiced (Eq. 4–5)",
            "Mask-based parallel training (Eq. 7–8)",
        ],
        "downsampling_rate": c.downsampling_rate,
        "bitrate_saving_pct": c.bitrate_saving_pct,
        "headline": headline_results(c),
    }


def headline_results(cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or VoCodecConfig()
    return {
        "libritts_kbps": c.libritts_target_kbps,
        "libritts_mushra": c.t1_mushra,
        "vctk_kbps": c.vctk_target_kbps,
        "vctk_mushra": c.t2_mushra,
        "bitrate_saving_vs_uniform_pct": c.bitrate_saving_pct,
        "beats_streamcodec_mushra": c.t1_mushra > c.streamcodec_t1_mushra,
    }


def table1_libritts() -> list[dict[str, Any]]:
    """Table 1 — LibriTTS 16 kHz @ 1.1 kbps."""
    c = VoCodecConfig()
    return [
        {
            "codec": "DAC",
            "streamable": False,
            "lsd": 0.936,
            "stoi": 0.888,
            "visqol": 3.781,
            "mushra": 74.83,
            "flops_g": 55.53,
            "params_m": 73.87,
        },
        {
            "codec": "BigCodec",
            "streamable": False,
            "lsd": 0.888,
            "stoi": 0.920,
            "visqol": 4.086,
            "mushra": c.bigcodec_t1_mushra,
            "flops_g": 61.03,
            "params_m": 159.32,
        },
        {
            "codec": "AudioDec",
            "streamable": True,
            "lsd": 0.988,
            "stoi": 0.698,
            "visqol": 3.617,
            "mushra": 71.02,
            "flops_g": 26.32,
            "params_m": 24.41,
        },
        {
            "codec": "MDCTCodec-S",
            "streamable": True,
            "lsd": 0.952,
            "stoi": 0.867,
            "visqol": 3.772,
            "mushra": 65.37,
            "flops_g": 2.32,
            "params_m": 6.75,
        },
        {
            "codec": "StreamCodec",
            "streamable": True,
            "lsd": c.streamcodec_t1_lsd,
            "stoi": c.streamcodec_t1_stoi,
            "visqol": c.streamcodec_t1_visqol,
            "mushra": c.streamcodec_t1_mushra,
            "flops_g": 2.32,
            "params_m": 6.75,
        },
        {
            "codec": "VoCodec",
            "streamable": True,
            "lsd": c.t1_lsd,
            "stoi": c.t1_stoi,
            "visqol": c.t1_visqol,
            "mushra": c.t1_mushra,
            "flops_g": c.t1_flops_g,
            "params_m": c.t1_params_m,
        },
    ]


def table2_vctk() -> list[dict[str, Any]]:
    """Table 2 — VCTK 48 kHz @ 2.7 kbps."""
    c = VoCodecConfig()
    return [
        {"codec": "DAC", "streamable": False, "lsd": 0.870, "stoi": 0.842, "visqol": 3.580, "mushra": 81.22},
        {"codec": "BigCodec", "streamable": False, "lsd": 0.850, "stoi": 0.880, "visqol": 3.678, "mushra": 84.73},
        {"codec": "AudioDec", "streamable": True, "lsd": 0.886, "stoi": 0.771, "visqol": 3.695, "mushra": 81.01},
        {"codec": "MDCTCodec-S", "streamable": True, "lsd": 0.862, "stoi": 0.836, "visqol": 3.714, "mushra": 79.30},
        {"codec": "StreamCodec", "streamable": True, "lsd": 0.855, "stoi": 0.842, "visqol": 3.802, "mushra": 79.38},
        {
            "codec": "VoCodec",
            "streamable": True,
            "lsd": c.t2_lsd,
            "stoi": c.t2_stoi,
            "visqol": c.t2_visqol,
            "mushra": c.t2_mushra,
        },
    ]


def table3_voicing_analysis() -> list[dict[str, Any]]:
    """Table 3 — voiced/unvoiced LSD @ LibriTTS 1.1 kbps."""
    c = VoCodecConfig()
    return [
        {
            "variant": "StreamCodec",
            "lsd": 0.918,
            "lsd_v": 0.740,
            "lsd_u": 0.620,
            "stoi": 0.896,
            "visqol": 4.048,
        },
        {
            "variant": "VoCodec",
            "lsd": c.t3_vocodec_lsd,
            "lsd_v": c.t3_vocodec_lsd_v,
            "lsd_u": c.t3_vocodec_lsd_u,
            "stoi": c.t3_vocodec_stoi,
            "visqol": c.t3_vocodec_visqol,
        },
        {
            "variant": "VoCodec-r",
            "lsd": c.t3_vocodec_r_lsd,
            "lsd_v": 0.796,
            "lsd_u": 0.622,
            "stoi": c.t3_vocodec_r_stoi,
            "visqol": 3.673,
        },
    ]


def fig2_abx_significant_wins() -> list[dict[str, Any]]:
    """Fig. 2 — ABX preference vs baselines @ 1.1 kbps (p < 0.05)."""
    return [
        {"baseline": "AudioDec", "vocodec_win_pct": 45.25, "p_value": 0.0298},
        {"baseline": "DAC", "vocodec_win_pct": 53.50, "p_value": 0.0071},
        {"baseline": "MDCTCodec-S", "vocodec_win_pct": 47.75, "p_value": 0.0071},
        {"baseline": "StreamCodec", "vocodec_win_pct": 52.50, "p_value": 0.0003},
    ]


def fig3_abx_vs_higher_bitrate() -> list[dict[str, Any]]:
    """Fig. 3 — VoCodec 1.1 kbps vs baselines @ 1.5 kbps (not significant)."""
    c = VoCodecConfig()
    return [
        {"baseline": "AudioDec @1.5 kbps", "p_value": 0.6788},
        {"baseline": "DAC @1.5 kbps", "p_value": 0.4638},
        {"baseline": "MDCTCodec-S @1.5 kbps", "p_value": 0.5686},
        {"baseline": "StreamCodec @1.5 kbps", "p_value": 0.3193},
        {"baseline": "SQCodec @1.5 kbps", "p_value": 0.1070},
        {"note": f"27% bitrate saving ({c.libritts_target_kbps} vs {c.uniform_comparison_kbps} kbps)"},
    ]


def benchmarks_bundle(cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or VoCodecConfig()
    return {
        "table1_libritts": table1_libritts(),
        "table2_vctk": table2_vctk(),
        "table3_voicing_analysis": table3_voicing_analysis(),
        "fig2_abx_significant": fig2_abx_significant_wins(),
        "fig3_abx_higher_bitrate": fig3_abx_vs_higher_bitrate(),
        "mushra_hidden_ref": c.mushra_hidden_ref,
        "mushra_anchor": c.mushra_anchor,
    }


def evaluation_demo(seed: int = 42, cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or VoCodecConfig()

    t = np.linspace(0, 0.5, int(c.libritts_sample_rate_hz * 0.5), endpoint=False)
    voiced_wave = 0.5 * np.sin(2 * np.pi * 120 * t)
    unvoiced_wave = rng.normal(0, 0.15, len(t))
    waveform = np.concatenate([voiced_wave, unvoiced_wave])

    flags = detect_voicing_flags(waveform, fs=c.libritts_sample_rate_hz, cfg=c)
    encoded = rng.normal(0, 1, (c.quantizer_dim, len(flags)))
    batch = mask_based_quantize_batch(
        encoded,
        np.array(flags, dtype=float),
        rng=rng,
    )

    kbps_libri = bitrate_kbps_simplified(
        fs=c.libritts_sample_rate_hz,
        voiced_ratio=c.libritts_voiced_ratio,
        cfg=c,
    )
    kbps_vctk = bitrate_kbps_simplified(
        fs=c.vctk_sample_rate_hz,
        voiced_ratio=c.vctk_voiced_ratio,
        cfg=c,
    )

    vocodec_row = next(r for r in table1_libritts() if r["codec"] == "VoCodec")
    stream_row = next(r for r in table1_libritts() if r["codec"] == "StreamCodec")

    return {
        "voicing_flags_sample": flags[:8],
        "voiced_token_layout": quantization_token_layout(1, c),
        "unvoiced_token_layout": quantization_token_layout(0, c),
        "libritts_bitrate_kbps": round(kbps_libri, 2),
        "vctk_bitrate_kbps": round(kbps_vctk, 2),
        "mask_quantize": {k: v for k, v in batch.items() if k != "quantized"},
        "beats_streamcodec_stoi": vocodec_row["stoi"] > stream_row["stoi"],
        "beats_streamcodec_visqol": vocodec_row["visqol"] > stream_row["visqol"],
        "vocodec_r_worse_than_vocodec": c.t3_vocodec_r_stoi < c.t3_vocodec_stoi,
    }


def pipeline_demo(seed: int = 42, cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or VoCodecConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
