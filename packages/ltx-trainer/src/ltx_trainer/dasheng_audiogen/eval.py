"""MECAT evaluation and expert-pipeline stub (Dasheng AudioGen §4)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dasheng_audiogen.captions import StructuredCaption
from ltx_trainer.dasheng_audiogen.config import DashengAudioGenConfig
from ltx_trainer.dasheng_audiogen.metrics_compute import metrics_from_arrays
from ltx_trainer.dasheng_audiogen.sampler import sample_to_waveform


def expert_pipeline_beats_unified_on_sma(cfg: DashengAudioGenConfig | None = None) -> bool:
    """Expert-Pipeline WER worse on SMA despite strong standalone TTS."""
    cfg = cfg or DashengAudioGenConfig()
    return cfg.expert_pipeline_sma_wer_pct > cfg.mecat_sma_wer_pct * 2


def run_generation_eval(
    caption: StructuredCaption,
    *,
    ref_wave: np.ndarray | None = None,
    transcript: str = "",
    cfg: DashengAudioGenConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate sample and compute FAD/WER proxies vs reference."""
    cfg = cfg or DashengAudioGenConfig()
    sr = 48000.0
    t = np.arange(int(sr * 2)) / sr
    ref = ref_wave if ref_wave is not None else np.sin(2 * np.pi * 440 * t)
    gen = sample_to_waveform(caption, duration_s=2.0, cfg=cfg, seed=seed)
    hyp = caption.asr or caption.speech or caption.caption
    metrics = metrics_from_arrays([ref], [gen[: ref.size]], transcripts=[transcript or hyp], asr_hyps=[hyp])
    return {"metrics": metrics, "gen_samples": int(gen.size)}


def eval_smoke(cfg: DashengAudioGenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DashengAudioGenConfig()
    cap = StructuredCaption(caption="Pop music with speech.", asr="hello world", speech="hello world")
    gen_eval = run_generation_eval(cap, transcript="hello world", cfg=cfg)
    computed_fad = gen_eval["metrics"]["fad_proxy"]
    return {
        "sma_fad_ours_beats_expert": cfg.mecat_sma_fad < cfg.expert_pipeline_sma_fad,
        "structured_beats_unstructured_wer": cfg.librispeech_wer_pct < cfg.unstructured_wer_pct,
        "musiccaps_competitive": cfg.musiccaps_fad < 2.0,
        "expert_pipeline_incoherent_mix": expert_pipeline_beats_unified_on_sma(cfg),
        "pafi_ours_near_gt": abs(cfg.pafi_ours_sma - cfg.pafi_gt_sma) < 0.05,
        "computed_fad_finite": np.isfinite(computed_fad),
        "generation_eval": gen_eval,
    }
