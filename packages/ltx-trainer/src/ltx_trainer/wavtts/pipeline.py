"""Framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.wavtts.config import WavTTSConfig
from ltx_trainer.wavtts.losses import flow_matching_x_loss, multi_scale_mel_stub
from ltx_trainer.wavtts.model import WavTTSStub
from ltx_trainer.wavtts.patchify import patchify_waveform
from ltx_trainer.wavtts.schedule import polyshift_schedule, sample_timestep_logit_normal, scale_waveform


def framework_card(cfg: WavTTSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WavTTSConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "github": cfg.github,
        "project_page": cfg.project_page,
        "paradigm": "flow matching + DiT on raw waveform patches",
        "patch_size": cfg.patch_size,
        "patch_rate_hz": cfg.patch_rate_hz,
        "params_M": cfg.params_m,
        "training_data": cfg.training_data,
        "objective": "x-prediction + multi-scale mel (λmel=0.05)",
        "noise_design": [
            f"variance alignment k={cfg.scale_k}",
            f"logit-normal t μ={cfg.logit_normal_mu}, σ={cfg.logit_normal_sigma}",
            f"PolyShift inference p={cfg.polyshift_p}, s={cfg.polyshift_s}",
        ],
        "inference": {"NFE": cfg.nfe, "CFG": cfg.cfg_scale},
    }


def table1_seed_tts() -> list[dict[str, Any]]:
    """Table 1 — Seed-TTS benchmark excerpt."""
    return [
        {"model": "F5-TTS", "en_WER": 1.65, "en_UTMOS": 3.73, "zh_CER": 1.55},
        {"model": "ZipVoice", "en_WER": 1.60, "en_UTMOS": 3.83, "zh_CER": 1.40},
        {"model": "LongCat-AudioDiT", "en_WER": 1.94, "en_SIM": 0.76, "zh_CER": 1.10},
        {"model": "WavTTS", "en_WER": 1.50, "en_SIM": 0.65, "en_UTMOS": 3.92, "zh_CER": 1.59, "zh_UTMOS": 3.08},
    ]


def table2_supervised_tts() -> list[dict[str, Any]]:
    """Table 2 — LJSpeech / LibriSpeech-PC."""
    return [
        {"model": "VITSLJ", "ljspeech_WER": 3.72, "librispeech_WER": 2.23},
        {"model": "JETS", "ljspeech_WER": 3.73, "librispeech_WER": 3.00},
        {"model": "WavTTS", "ljspeech_WER": 3.43, "ljspeech_UTMOS": 4.39, "librispeech_WER": 2.02, "librispeech_UTMOS": 4.36},
    ]


def table3_objective_ablation() -> list[dict[str, Any]]:
    """Table 3 — x-prediction vs v-prediction / λmel."""
    return [
        {"target": "v-prediction", "lambda_mel": 0.05, "WER": 1.67, "SIM": 0.61, "UTMOS": 3.94},
        {"target": "x-prediction", "lambda_mel": 0.0, "WER": 1.92, "SIM": 0.56, "UTMOS": 3.77},
        {"target": "x-prediction", "lambda_mel": 0.05, "WER": 1.65, "SIM": 0.65, "UTMOS": 3.93},
    ]


def forward_smoke(cfg: WavTTSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WavTTSConfig()
    model = WavTTSStub(cfg)
    b, t_len = cfg.demo_batch, cfg.demo_time_samples
    x1 = torch.randn(b, t_len) * 0.12
    x1_scaled = scale_waveform(x1, cfg.scale_k)
    x0 = torch.randn_like(x1_scaled)
    timesteps = sample_timestep_logit_normal(b, mu=cfg.logit_normal_mu, sigma=cfg.logit_normal_sigma)
    xt = (1.0 - timesteps).view(-1, 1) * x0 + timesteps.view(-1, 1) * x1_scaled
    mask = torch.zeros(b, t_len)
    mask[:, t_len // 4 : 3 * t_len // 4] = 1.0
    x_ctx = x1_scaled * (1.0 - mask)
    x_pred = model(xt, timesteps, x_ctx, mask=mask)
    pred_p = patchify_waveform(x_pred, cfg.patch_size)
    target_p = patchify_waveform(x1_scaled, cfg.patch_size)
    mask_p = patchify_waveform(mask, cfg.patch_size).amax(dim=-1, keepdim=True)
    l_fm = flow_matching_x_loss(pred_p, target_p, mask_p, timesteps, t_clip_max=cfg.t_clip_max)
    l_mel = multi_scale_mel_stub(x_pred / cfg.scale_k, x1, mask)
    sched = polyshift_schedule(cfg.nfe, p=cfg.polyshift_p, s=cfg.polyshift_s)
    return {
        "loss_fm": float(l_fm.detach()),
        "loss_mel": float(l_mel.detach()),
        "pred_shape": list(x_pred.shape),
        "polyshift_steps": len(sched) - 1,
        "polyshift_t0": float(sched[0]),
        "polyshift_t_end": float(sched[-1]),
    }


def evaluation_demo(cfg: WavTTSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WavTTSConfig()
    return {
        "framework": framework_card(cfg),
        "table1": table1_seed_tts(),
        "table2": table2_supervised_tts(),
        "table3": table3_objective_ablation(),
        "headline": {
            "seed_en_WER": cfg.seed_en_wer,
            "seed_en_UTMOS": cfg.seed_en_utmos,
            "best_en_WER_note": "best among NAR baselines in Table 1",
        },
        "forward": forward_smoke(cfg),
    }
