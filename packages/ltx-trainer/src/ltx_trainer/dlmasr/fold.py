"""Fold DLM-ASR decoding proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dlmasr.config import DecodingStrategy, DlmAsrConfig
from ltx_trainer.dlmasr.decoding import block_decode_smoke


def dlmasr_meta_block() -> dict[str, Any]:
    cfg = DlmAsrConfig()
    return {
        "dlmasr": {
            "arxiv_id": "2605.29613",
            "task": "dlm_parallel_asr_decoding",
            "fold_role": "asr_decode_sidecar",
            "baseline": cfg.baseline_system,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(dlmasr_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    if arr.size < 1600:
        return out

    est_tokens = max(8, min(64, int(arr.size / 800)))
    static = block_decode_smoke(est_tokens, strategy=DecodingStrategy.STATIC_THRESHOLD, seed=42)
    out["dlmasr"].update(
        {
            "est_text_tokens": est_tokens,
            "static_decode_rounds_proxy": static["rounds"],
            "recommended_strategy": "static_threshold",
        }
    )
    return out
