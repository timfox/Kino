"""Framework card, Tables 1–6, 5, 9, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.clean_codec.config import CleanCodecConfig
from ltx_trainer.clean_codec.losses import combined_codec_loss
from ltx_trainer.clean_codec.model import CleanCodecStub
from ltx_trainer.clean_codec.quantize import codebook_size


def framework_card(cfg: CleanCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CleanCodecConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "token_rates_tps": list(cfg.token_rates_tps),
        "fsq_levels": list(cfg.fsq_levels),
        "codebook_size": codebook_size(cfg.fsq_levels),
        "global_emb_dim": cfg.global_emb_dim,
        "training": {
            "speech_enhancement": True,
            "ssl_conditioning": cfg.ssl_model,
            "global_conditioning": cfg.sv_model,
            "two_stage": True,
            "hours": 2400,
        },
        "params_m": {
            "autoencoder": cfg.autoencoder_params_m,
            "vocoder": cfg.vocoder_params_m,
            "inference_ae": cfg.autoencoder_params_m + cfg.vocoder_params_m - 47,
        },
    }


def table1_librispeech_test_clean() -> list[dict[str, Any]]:
    """Table 1 excerpt — low token rate baselines vs CleanCodec@12.5."""
    return [
        {"codec": "WavTokenizer", "tps": 40, "WER": 9.0, "CER": 5.0, "SIM": 0.62},
        {"codec": "FocalCodec", "tps": 12.5, "WER": 8.3, "CER": 4.5, "SIM": 0.45},
        {"codec": "Kanade", "tps": 12.5, "WER": 4.0, "CER": 2.1, "SIM": 0.65},
        {"codec": "CleanCodec@12.5", "tps": 12.5, "WER": 2.7, "CER": 1.4, "SIM": 0.86},
    ]


def table2_ood_excerpt() -> list[dict[str, Any]]:
    """Table 2 — OOD @12.5."""
    return [
        {"dataset": "Expresso", "codec": "Kanade", "WER": 9.3, "SIM": 0.55},
        {"dataset": "Expresso", "codec": "CleanCodec@12.5", "WER": 3.9, "SIM": 0.82},
        {"dataset": "AISHELL-3", "codec": "Kanade", "CER": 7.5, "SIM": 0.47},
        {"dataset": "AISHELL-3", "codec": "CleanCodec@12.5", "CER": 1.5, "SIM": 0.84},
    ]


def table3_disentanglement() -> list[dict[str, Any]]:
    """Table 3 — SV / ASR on disentangled tokens."""
    return [
        {"codec": "Kanade", "tps": 12.5, "SV_ACC": 96.82, "SV_EER": 3.38, "ASR_WER": 8.1},
        {"codec": "CleanCodec@12.5", "tps": 12.5, "SV_ACC": 99.92, "SV_EER": 0.58, "ASR_WER": 5.6},
    ]


def table5_tts_seed() -> list[dict[str, Any]]:
    """Table 5 — Seed-TTS-eval."""
    return [
        {"codec": "Qwen3", "tps": 200, "train_hhmm": "10:02", "RTF": 2.930, "SIM": 0.36, "WER": 9.1},
        {"codec": "Kanade", "tps": 12.5, "train_hhmm": "00:59", "RTF": 0.169, "SIM": 0.45, "WER": 5.6},
        {"codec": "CleanCodec", "tps": 12.5, "train_hhmm": "01:00", "RTF": 0.170, "SIM": 0.56, "WER": 3.9},
    ]


def table6_ablation() -> list[dict[str, Any]]:
    """Table 6 — design ablations @12.5 test-clean."""
    return [
        {"variant": "Baseline", "WER": 2.7, "SIM": 0.86},
        {"variant": "One-stage training", "WER": 4.1, "SIM": 0.80},
        {"variant": "No SSL cond.", "WER": 8.1, "SIM": 0.84},
        {"variant": "No global cond.", "WER": 2.8, "SIM": 0.74},
        {"variant": "No denoise", "WER": 3.4, "SIM": 0.82},
    ]


def table9_codec_rtf() -> list[dict[str, Any]]:
    """Table 9 — encode/decode RTF."""
    return [
        {"codec": "Kanade", "total_RTF": 0.0056, "RTFx": 177},
        {"codec": "CleanCodec@12.5", "total_RTF": 0.0045, "RTFx": 221},
    ]


def forward_smoke(cfg: CleanCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CleanCodecConfig()
    model = CleanCodecStub(cfg)
    mel = torch.randn(2, cfg.demo_mel_bins, cfg.demo_mel_frames)
    out = model(mel)
    m_hat = out["m_hat"]
    if m_hat.shape[1] != cfg.demo_mel_bins:
        m_hat = m_hat.transpose(1, 2)
    t = min(mel.shape[-1], m_hat.shape[-1])
    m_tgt = mel[..., :t]
    m_pred = m_hat[..., :t] if m_hat.shape[-1] >= t else m_hat.transpose(1, 2)[..., :t]
    if m_pred.dim() == 3 and m_pred.shape[1] != cfg.demo_mel_bins:
        m_pred = m_pred.transpose(1, 2)
    losses = combined_codec_loss(
        m_tgt,
        m_pred,
        out["z_q"],
        out["s_hat"],
        torch.randn(2, cfg.global_emb_dim, device=mel.device),
        out["g_hat"],
    )
    row = [r for r in table1_librispeech_test_clean() if r["codec"] == "CleanCodec@12.5"][0]
    return {
        "loss_total": float(losses["total"].detach()),
        "z_q_shape": list(out["z_q"].shape),
        "codebook_size": codebook_size(cfg.fsq_levels),
        "table1_wer": row["WER"],
        "table1_sim": row["SIM"],
    }


def evaluation_demo(cfg: CleanCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CleanCodecConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_librispeech_test_clean(),
        "table2_ood": table2_ood_excerpt(),
        "table3_disentangle": table3_disentanglement(),
        "table5_tts": table5_tts_seed(),
        "table6_ablation": table6_ablation(),
        "table9_rtf": table9_codec_rtf(),
        "forward": forward_smoke(cfg),
    }
