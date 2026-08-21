"""Fold MUSTBENCH temporal-grounding proxies into LTX audio sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.mustbench.config import MustBenchConfig
from ltx_trainer.mustbench.metrics import hit_at_t


def mustbench_meta_block() -> dict[str, Any]:
    cfg = MustBenchConfig()
    return {
        "mustbench": {
            "arxiv_id": "2605.29300",
            "task": "music_temporal_grounding",
            "fold_role": "music_audio_sidecar",
            "num_tasks": 5,
            "hit_tolerance_s": cfg.hit_tolerance_s,
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(mustbench_meta_block())

    wave = data.get("waveform") or data.get("audio")
    if wave is None:
        return out

    arr = np.asarray(wave, dtype=np.float64).ravel()
    if arr.size < 1600:
        return out

    duration_s = arr.size / 16000.0
    # Proxy: estimate onset from energy rise
    frame = max(512, arr.size // 64)
    energy = [float(np.mean(arr[i : i + frame] ** 2)) for i in range(0, arr.size - frame, frame)]
    if len(energy) < 2:
        return out
    onset_idx = int(np.argmax(np.diff(energy)))
    onset_s = onset_idx * frame / 16000.0
    hit = hit_at_t([onset_s], [onset_s], tolerance_s=3.0)

    out["mustbench"].update(
        {
            "est_duration_s": round(duration_s, 2),
            "tsg_onset_proxy_s": round(onset_s, 2),
            "tsg_self_hit3": hit,
        }
    )
    return out
