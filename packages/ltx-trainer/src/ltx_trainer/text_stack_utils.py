"""Gemma 4 ↔ LTX text-stack helpers: caption indexing, checkpoint sidecars, bridge folding."""

from __future__ import annotations

import hashlib
import json
import os
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


def lookup_caption(
    caption_index: dict[str, str],
    shard_path: str | Path,
    *,
    latents_root: str | Path | None = None,
) -> str | None:
    """Resolve a manifest caption for a latent shard path (backfill / audit helpers)."""
    path = Path(shard_path)
    candidates = caption_keys_for_tensor_path(path)
    if latents_root is not None:
        try:
            rel = path.relative_to(Path(latents_root).resolve())
            candidates = caption_keys_for_tensor_path(rel) + candidates
        except ValueError:
            pass
    for key in candidates:
        cap = caption_index.get(key)
        if isinstance(cap, str) and cap.strip():
            return cap.strip()
    return None


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


def apply_low_rank_bridge_to_aggregate(
    aggregate_weight: Tensor,
    weight_out: Tensor,
    weight_in: Tensor,
) -> Tensor:
    """``aggregate @ weight_out @ weight_in`` without building the dense bridge (~250 GiB for rank-512)."""
    agg = aggregate_weight.float()
    return (agg @ weight_out.float()) @ weight_in.float()


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
        return apply_low_rank_bridge_to_aggregate(
            aggregate_weight,
            layers[1].weight,
            layers[0].weight,
        )
    if isinstance(bridge, nn.Linear):
        b_full = bridge.weight
    else:
        raise TypeError(f"Unsupported bridge type: {type(bridge)}")

    if b_full.shape != (ckpt_flat, gemma_flat_in):
        raise ValueError(f"Bridge shape {tuple(b_full.shape)} expected ({ckpt_flat}, {gemma_flat_in})")

    return aggregate_weight.float() @ b_full.float()


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
    if flat is not None:
        return int(flat)
    # LTX-2.5 split transformers ship without text aggregates; official flat is still 188160.
    fallback = os.environ.get("GOPEX_LTX_OFFICIAL_FLAT_DIM", "").strip()
    if fallback:
        return int(fallback)
    raise ValueError(
        f"Could not read video_aggregate_embed.in_features from {model_path}. "
        "For LTX-2.5 transformers without aggregates, set GOPEX_LTX_OFFICIAL_FLAT_DIM=188160 "
        "and pass --inject-aggregates to fold_flat_dim_bridge.py."
    )


def gemma_caption_cache_enabled() -> bool:
    """Default on for phase1a-style training (set GOPEX_GEMMA_CAPTION_CACHE=0 to disable)."""
    return os.environ.get("GOPEX_GEMMA_CAPTION_CACHE", "1").strip().lower() not in ("0", "false", "no")


def gemma_caption_cache_key(caption: str) -> str:
    return hashlib.sha256(caption.strip().encode("utf-8")).hexdigest()[:32]


def gemma_caption_cache_path(cache_dir: Path, caption: str) -> Path:
    return cache_dir / f"{gemma_caption_cache_key(caption)}.pt"


def load_gemma_caption_cache(cache_dir: Path, caption: str) -> tuple[tuple[Tensor, ...] | Tensor, Tensor] | None:
    path = gemma_caption_cache_path(cache_dir, caption)
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=True)
    hs = blob["hidden_states"]
    if isinstance(hs, list):
        hs = tuple(hs)
    return hs, blob["prompt_mask"]


def save_gemma_caption_cache(
    cache_dir: Path,
    caption: str,
    hidden_states: tuple[Tensor, ...] | Tensor,
    prompt_mask: Tensor,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = gemma_caption_cache_path(cache_dir, caption)
    hs = list(hidden_states) if isinstance(hidden_states, tuple) else hidden_states
    torch.save({"hidden_states": hs, "prompt_mask": prompt_mask.detach().cpu()}, path)
    return path


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
