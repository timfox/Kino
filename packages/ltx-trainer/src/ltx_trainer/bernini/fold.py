"""Fold Bernini semantic segment plan metadata into video latent shards (arXiv:2605.22344)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.bernini.planner import inference_mask_ratio, train_mask_ratio_beta, visible_token_fraction


def bernini_meta_block() -> dict[str, Any]:
    return {
        "bernini": {
            "arxiv_id": "2605.22344",
            "fold_role": "latent_semantic_segment_plan",
        }
    }


def segment_ids_for_frames(num_frames: int, *, num_segments: int = 3) -> list[int]:
    if num_frames <= 0:
        return []
    return [i % num_segments for i in range(num_frames)]


def _infer_num_frames(data: dict[str, Any], arr: np.ndarray | None) -> int:
    nf = data.get("num_frames")
    if isinstance(nf, int) and nf > 0:
        return nf
    if arr is None:
        return 0
    if arr.ndim == 5:
        return int(arr.shape[1]) if arr.shape[1] <= 256 else int(arr.shape[0])
    if arr.ndim == 4:
        return int(arr.shape[0]) if arr.shape[0] <= 256 else int(arr.shape[1])
    return 0


def annotate_video_latent_data(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.update(bernini_meta_block())
    latents = data.get("latents")
    arr = None
    if latents is not None:
        arr = np.asarray(latents.detach().cpu().float().numpy() if hasattr(latents, "detach") else latents)
    n = _infer_num_frames(data, arr)
    num_segments = min(6, max(2, (n // 8) or 2))
    seg_ids = segment_ids_for_frames(n, num_segments=num_segments)
    train_mr = train_mask_ratio_beta()
    infer_mr = inference_mask_ratio(0, 8)
    out["bernini"].update(
        {
            "num_frames": n,
            "num_segments": num_segments,
            "segment_ids": seg_ids[:32],
            "train_mask_ratio": round(train_mr, 4),
            "inference_mask_ratio_k0": round(infer_mr, 4),
            "visible_token_fraction": round(visible_token_fraction(infer_mr), 4),
        }
    )
    return out
