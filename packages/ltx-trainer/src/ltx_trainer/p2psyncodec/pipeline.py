"""Framework card, paper tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.p2psyncodec.config import P2PSynCodecConfig
from ltx_trainer.p2psyncodec.quantizer import (
    bitrate_kbps,
    codebook_lookup,
    plain_vq_token,
    pseudo_vq_predict_logits,
    pseudo_vq_token,
    synergistic_quantized_vector,
    teacher_forcing_train_step,
)


def framework_card(cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or P2PSynCodecConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "ultra_low_bitrate_neural_speech_codec",
        "components": [
            "MDCT ConvNeXt v2 encoder-decoder",
            "Plain VQ (transmitted tokens, Eq. 2 bitrate)",
            "N pseudo VQs (predicted auxiliary tokens, zero bitrate)",
            "Two-stage training: RVQ teacher → pseudo-VQ CE (Eq. 6–7)",
        ],
        "pseudo_vq_count": c.pseudo_vq_count,
        "bitrate_saving_pct": c.bitrate_saving_pct,
        "headline": headline_results(c),
    }


def headline_results(cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or P2PSynCodecConfig()
    return {
        "libritts_kbps": c.libritts_target_kbps,
        "libritts_utmos": c.t1_utmos,
        "vctk_kbps": c.vctk_target_kbps,
        "comparison_baseline_kbps": c.comparison_baseline_kbps,
        "bitrate_saving_pct": c.bitrate_saving_pct,
        "beats_mdctcodec_utmos": c.t1_utmos > c.mdctcodec_t1_utmos,
        "comparable_bigcodec_utmos": abs(c.t1_utmos - c.bigcodec_t1_utmos) < 0.02,
    }


def table1_objective() -> list[dict[str, Any]]:
    """Table 1 — ultra-low bitrate objective results."""
    c = P2PSynCodecConfig()
    return [
        {
            "codec": "MDCTCodec",
            "libritts_utmos": c.mdctcodec_t1_utmos,
            "libritts_stoi": c.mdctcodec_t1_stoi,
            "libritts_visqol": c.mdctcodec_t1_visqol,
            "vctk_sigmos": 2.846,
            "flops_g": c.mdctcodec_t1_flops_g,
            "params_m": c.mdctcodec_t1_params_m,
        },
        {
            "codec": "DAC",
            "libritts_utmos": 2.725,
            "libritts_stoi": 0.818,
            "libritts_visqol": 3.386,
            "vctk_sigmos": 2.971,
            "flops_g": 55.53,
            "params_m": 73.87,
        },
        {
            "codec": "BigCodec",
            "libritts_utmos": c.bigcodec_t1_utmos,
            "libritts_stoi": 0.872,
            "libritts_visqol": 3.682,
            "vctk_sigmos": 3.277,
            "flops_g": c.bigcodec_t1_flops_g,
            "params_m": c.bigcodec_t1_params_m,
        },
        {
            "codec": "WavTokenizer",
            "libritts_utmos": c.wavtokenizer_t1_utmos,
            "libritts_stoi": 0.834,
            "libritts_visqol": 3.484,
            "vctk_sigmos": 3.232,
            "flops_g": 4.21,
            "params_m": 71.65,
        },
        {
            "codec": "P2PSynCodec",
            "libritts_utmos": c.t1_utmos,
            "libritts_stoi": c.t1_stoi,
            "libritts_visqol": c.t1_visqol,
            "vctk_sigmos": c.t1_sigmos,
            "flops_g": c.t1_flops_g,
            "params_m": c.t1_params_m,
        },
    ]


def table2_pseudo_vq_ablation() -> list[dict[str, Any]]:
    """Table 2 — impact of pseudo VQ count N on LibriTTS."""
    return [
        {"n_pseudo": 1, "all_utmos": 3.787, "plain_utmos": 3.048, "all_stoi": 0.845, "all_visqol": 3.551},
        {"n_pseudo": 3, "all_utmos": 3.947, "plain_utmos": 2.324, "all_stoi": 0.823, "all_visqol": 3.476},
        {"n_pseudo": 5, "all_utmos": 3.986, "plain_utmos": 1.943, "all_stoi": 0.798, "all_visqol": 3.208},
        {"n_pseudo": 7, "all_utmos": 3.889, "plain_utmos": 1.296, "all_stoi": 0.725, "all_visqol": 2.761},
    ]


def fig3_abx_vs_high_bitrate() -> list[dict[str, Any]]:
    """Fig. 3 — P2PSynCodec @0.5 kbps vs higher-bitrate baselines (not significant)."""
    c = P2PSynCodecConfig()
    return [
        {"baseline": "MDCTCodec @2.0 kbps", "p_value": 0.2823},
        {"baseline": "DAC @2.0 kbps", "p_value": 0.8890},
        {"baseline": "WavTokenizer @2.0 kbps", "p_value": 0.3725},
        {"baseline": "SQCodec @1.5 kbps", "p_value": 0.3285},
        {"note": f"75% bitrate saving ({c.libritts_target_kbps} vs {c.comparison_baseline_kbps} kbps)"},
    ]


def benchmarks_bundle(cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or P2PSynCodecConfig()
    return {
        "table1_objective": table1_objective(),
        "table2_pseudo_vq_ablation": table2_pseudo_vq_ablation(),
        "fig3_abx_high_bitrate": fig3_abx_vs_high_bitrate(),
        "optimal_n_pseudo": c.pseudo_vq_count,
    }


def evaluation_demo(seed: int = 42, cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    c = cfg or P2PSynCodecConfig()

    codebook = rng.normal(0, 1, (c.plain_codebook_size, c.code_vector_dim))
    encoded = rng.normal(0, 1, c.code_vector_dim)
    d_pl = plain_vq_token(encoded, codebook)
    plain_emb = codebook_lookup(codebook, d_pl)

    pseudo_embs: list[np.ndarray] = []
    for _ in range(c.pseudo_vq_count):
        logits = pseudo_vq_predict_logits(
            plain_embedding=plain_emb,
            prior_embeddings=pseudo_embs,
            rng=rng,
        )
        token = pseudo_vq_token(logits)
        pseudo_embs.append(codebook_lookup(codebook, token % c.plain_codebook_size))

    e_hat = synergistic_quantized_vector(plain_emb, pseudo_embs)
    teacher_tokens = [rng.integers(0, c.plain_codebook_size) for _ in range(c.pseudo_vq_count + 1)]
    train = teacher_forcing_train_step(
        teacher_tokens,
        [codebook] * (c.pseudo_vq_count + 1),
        n_pseudo=c.pseudo_vq_count,
        rng=rng,
    )

    kbps_16k = bitrate_kbps(fs=c.libritts_sample_rate_hz, cfg=c)
    kbps_48k = bitrate_kbps(fs=c.vctk_sample_rate_hz, cfg=c)
    p2p_row = next(r for r in table1_objective() if r["codec"] == "P2PSynCodec")
    mdct_row = next(r for r in table1_objective() if r["codec"] == "MDCTCodec")

    t2 = table2_pseudo_vq_ablation()
    peak_n = max(t2, key=lambda r: r["all_utmos"])

    return {
        "plain_token": d_pl,
        "pseudo_token_count": c.pseudo_vq_count,
        "quantized_norm": float(np.linalg.norm(e_hat)),
        "libritts_bitrate_kbps": round(kbps_16k, 2),
        "vctk_bitrate_kbps": round(kbps_48k, 2),
        "teacher_forcing_mean_ce": train["mean_ce"],
        "beats_mdctcodec_utmos": p2p_row["libritts_utmos"] > mdct_row["libritts_utmos"],
        "optimal_n_pseudo": c.pseudo_vq_count,
        "peak_all_utmos_n": peak_n["n_pseudo"],
        "plain_only_degrades_with_n": t2[-1]["plain_utmos"] < t2[1]["plain_utmos"],
    }


def pipeline_demo(seed: int = 42, cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or P2PSynCodecConfig()
    return {
        "framework": framework_card(c),
        "benchmarks": benchmarks_bundle(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
    }
