"""Fold LuckyHDR bracket merge readiness into video / HDR latent shards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.hdr_ingest import lucky_hdr_meta_block
from ltx_trainer.lucky_hdr.capture_ae import bracket_evs


def _latent_time_series(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D or 5D latents, got shape {arr.shape}")
    if arr.shape[0] <= arr.shape[1] and arr.shape[0] <= 64:
        return arr
    if arr.shape[1] <= 64:
        return np.transpose(arr, (1, 0, 2, 3))
    return arr


def _bracket_stack_stats(stack: np.ndarray) -> tuple[float, int]:
    """``stack`` ``[N,C,F,H,W]`` or ``[N,C,H,W]`` → mean inter-bracket diff, N."""
    if stack.ndim == 5:
        n = int(stack.shape[0])
        diffs = [float(np.abs(stack[i] - stack[i + 1]).mean()) for i in range(n - 1)]
        spread = float(np.mean(diffs)) if diffs else 0.0
        return spread, n
    if stack.ndim == 4:
        n = int(stack.shape[0])
        diffs = [float(np.abs(stack[i] - stack[i + 1]).mean()) for i in range(n - 1)]
        spread = float(np.mean(diffs)) if diffs else 0.0
        return spread, n
    return 0.5, 1


def _handheld_shake_proxy(z: np.ndarray) -> float:
    z = _latent_time_series(z)
    if z.shape[0] < 2:
        return 0.5
    frame_energy = z.reshape(z.shape[0], -1).mean(axis=1)
    return float(np.mean(np.abs(np.diff(frame_energy))))


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    ev_stack = data.get("hdr_ldr_ev_stack")
    hdr_meta = data.get("hdr_meta") if isinstance(data.get("hdr_meta"), dict) else {}
    ev_list = data.get("hdr_ev_list")

    num_brackets = 5
    if ev_list is not None:
        num_brackets = len(ev_list) if hasattr(ev_list, "__len__") else num_brackets
    elif ev_stack is not None:
        arr0 = np.asarray(
            ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack
        )
        num_brackets = int(arr0.shape[0]) if arr0.ndim >= 1 else num_brackets

    block = lucky_hdr_meta_block(num_frames=num_brackets)
    out.update(block)

    merge_readiness = 0.65
    bracket_spread = 0.5
    if ev_stack is not None:
        arr = np.asarray(
            ev_stack.detach().cpu().float().numpy() if hasattr(ev_stack, "detach") else ev_stack
        )
        bracket_spread, num_brackets = _bracket_stack_stats(arr)
        merge_readiness = float(np.clip(1.0 - bracket_spread * 2.5, 0.0, 1.0))
    else:
        latents = data.get("latents")
        if latents is not None:
            arr = np.asarray(
                latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents
            )
            shake = _handheld_shake_proxy(arr)
            merge_readiness = float(np.clip(1.0 - shake * 3.0, 0.0, 1.0))
            bracket_spread = shake

    evs = bracket_evs(n=num_brackets, span=2.0) if num_brackets > 1 else []
    out["lucky_hdr"].update(
        {
            "merge_readiness_proxy": round(merge_readiness, 4),
            "bracket_spread_proxy": round(bracket_spread, 5),
            "mos_proxy": round(1.0 + merge_readiness * 4.0, 3),
            "has_ev_stack": ev_stack is not None,
            "bracket_evs": evs[:num_brackets],
            "hdr_vae_encoding": hdr_meta.get("hdr_vae_encoding"),
        }
    )
    return out
