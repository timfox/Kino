"""Gemma 4 ↔ LTX text-stack helpers: caption indexing, checkpoint sidecars, bridge folding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ltx_core.loader.helpers import peek_video_aggregate_embed_in_features


def load_caption_index(manifest_path: str | Path, *, caption_key: str = "caption") -> dict[str, str]:
    """Map manifest media stem → caption (e.g. ``clips/foo`` from ``clips/foo.mp4``)."""
    path = Path(manifest_path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {path}")
    index: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        media = entry.get("media_path")
        cap = entry.get(caption_key)
        if not isinstance(media, str) or not isinstance(cap, str):
            continue
        key = str(Path(media.strip()).with_suffix(""))
        index[key] = cap.strip()
    if not index:
        raise ValueError(f"No captions indexed from {path}")
    return index


def caption_keys_for_tensor_path(file_rel_path: str | Path) -> list[str]:
    """Candidate manifest keys for a precomputed shard path (most specific first)."""
    p = Path(file_rel_path)
    stem = str(p.with_suffix(""))
    keys = [stem, str(Path(p.name).with_suffix(""))]
    for prefix in ("latents/", "conditions/", "latent_conditions/"):
        if stem.startswith(prefix):
            keys.append(stem[len(prefix) :])
    # legacy latent_X.pt → condition_X
    if p.name.startswith("latent_"):
        keys.append(f"condition_{p.stem[7:]}")
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def caption_key_for_tensor_path(file_rel_path: str | Path) -> str:
    """Primary lookup key (first candidate from :func:`caption_keys_for_tensor_path`)."""
    return caption_keys_for_tensor_path(file_rel_path)[0]


def feature_extractor_state_dict(fe: nn.Module, *, prefix: str = "feature_extractor.") -> dict[str, Tensor]:
    """Export trainable feature-extractor weights (bridge + aggregate linears)."""
    return {f"{prefix}{k}": v.detach() for k, v in fe.state_dict().items()}


def load_feature_extractor_state_dict(fe: nn.Module, sd: dict[str, Tensor], *, prefix: str = "feature_extractor.") -> None:
    local = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    if local:
        fe.load_state_dict(local, strict=False)


def compose_low_rank_bridge_weights(
    weight_out: Tensor,
    weight_in: Tensor,
) -> Tensor:
    """``Linear(rank→out).weight @ Linear(in→rank).weight`` → dense ``[out, in]``."""
    return weight_out @ weight_in


def fold_aggregate_with_bridge(
    aggregate_weight: Tensor,
    bridge: nn.Module | None,
    *,
    gemma_flat_in: int,
) -> Tensor:
    """Fold bridge into ``video_aggregate_embed.weight`` → ``[out_features, gemma_flat_in]``.

    ``aggregate_weight`` is ``[out, ckpt_flat]`` from the pretrained LTX checkpoint.
    """
    out_dim, ckpt_flat = aggregate_weight.shape
    if bridge is None:
        if ckpt_flat == gemma_flat_in:
            return aggregate_weight
        raise ValueError(
            f"No flat_dim_bridge on feature_extractor but aggregate in_features={ckpt_flat} "
            f"!= Gemma flat {gemma_flat_in}"
        )

    if isinstance(bridge, nn.Sequential):
        layers = [m for m in bridge.children() if isinstance(m, nn.Linear)]
        if len(layers) != 2:
            raise ValueError("Expected Sequential flat_dim_bridge with two Linear layers")
        b_full = compose_low_rank_bridge_weights(layers[1].weight, layers[0].weight)
    elif isinstance(bridge, nn.Linear):
        b_full = bridge.weight
    else:
        raise TypeError(f"Unsupported bridge type: {type(bridge)}")

    if b_full.shape != (ckpt_flat, gemma_flat_in):
        raise ValueError(f"Bridge shape {tuple(b_full.shape)} expected ({ckpt_flat}, {gemma_flat_in})")

    return aggregate_weight @ b_full


def gemma_flat_dim_from_path(gemma_path: str | Path) -> int:
    """``hidden_size * (num_hidden_layers + 1)`` from Gemma 4 ``config.json``."""
    root = Path(gemma_path).expanduser().resolve()
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        for c in root.glob("**/config.json"):
            try:
                raw = json.loads(c.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if raw.get("model_type") == "gemma4":
                cfg_path = c
                break
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = raw["text_config"]
    return int(tc["hidden_size"]) * (int(tc["num_hidden_layers"]) + 1)


def ltx_checkpoint_flat_dim(model_path: str | Path) -> int:
    flat = peek_video_aggregate_embed_in_features(str(model_path))
    if flat is None:
        raise ValueError(f"Could not read video_aggregate_embed.in_features from {model_path}")
    return int(flat)


def fold_metadata_summary(
    *,
    gemma_flat: int,
    ckpt_flat_before: int,
    out_features: int,
) -> dict[str, Any]:
    return {
        "fold_gemma_flat_in": gemma_flat,
        "fold_ckpt_flat_before": ckpt_flat_before,
        "fold_video_aggregate_out_features": out_features,
        "fold_native_geometry": gemma_flat,
    }
