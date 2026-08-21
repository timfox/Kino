"""Shared model-construction helpers used by both SingleGPUModelBuilder and StreamingModelBuilder."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator

_M = TypeVar("_M", bound=nn.Module)


def _read_native_manifest(checkpoint_dir: Path) -> dict | None:
    manifest_path = checkpoint_dir / "native_manifest.json"
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _ltx_ckpt_from_env() -> Path | None:
    for key in ("LTX_CKPT", "GOPEX_UPSTREAM_LTX_DEV"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
    return None


def load_state_dict(
    paths: str | tuple[str, ...] | list[str],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device | None,
    sd_ops: SDOps | None = None,
) -> StateDict:
    """Load a state dict from disk, using registry caching."""
    if isinstance(paths, str):
        path_list = [paths]
    elif isinstance(paths, tuple):
        path_list = list(paths)
    else:
        path_list = paths
    cached = registry.get(path_list, sd_ops)
    if cached is not None:
        return cached
    result = loader.load(path_list, sd_ops=sd_ops, device=device)
    registry.add(path_list, sd_ops=sd_ops, state_dict=result)
    return result


def read_model_config(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read metadata from the first shard of a checkpoint."""
    first = model_path[0] if isinstance(model_path, tuple) else model_path
    return loader.metadata(first)


def read_ltx_checkpoint_config(checkpoint_path: str | Path, loader: StateDictLoader) -> dict:
    """Read LTX schema config from safetensors metadata or a native fold manifest."""
    path = Path(checkpoint_path).expanduser().resolve()
    manifest = _read_native_manifest(path) if path.is_dir() else None
    embedded = manifest.get("ltx_config") if manifest else None
    if isinstance(embedded, dict) and embedded:
        return embedded

    if path.is_file():
        cfg = read_model_config(str(path), loader)
        if cfg:
            return cfg

    fallback: Path | None = None
    if manifest:
        for key in ("source_ltx_ckpt", "vae_checkpoint"):
            raw = manifest.get(key)
            if raw:
                candidate = Path(str(raw)).expanduser()
                if candidate.is_file():
                    fallback = candidate
                    break
    if fallback is None:
        fallback = _ltx_ckpt_from_env()
    if fallback is not None:
        return read_model_config(str(fallback), loader)
    return {}


def resolve_ltx_checkpoint_paths(checkpoint_path: str | Path) -> tuple[str, ...]:
    """Resolve native-fold or sharded safetensors paths for model builders."""
    from ltx_core.block_streaming.builder import expand_ltx_checkpoint_safetensors_paths

    return expand_ltx_checkpoint_safetensors_paths(str(checkpoint_path))


def _vae_path_from_env() -> Path | None:
    """Standalone video VAE (LTX-2.5 split assets) via preprocess / infer env."""
    for key in ("GOPEX_PREP_VAE_CKPT", "GOPEX_VIDEO_VAE"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path
    return None


def resolve_audio_vae_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Standalone audio VAE (LTX-2.5 split) or unified 2.3 checkpoint.

    Video ConvVAE files do not contain audio encoder weights. Prefer
    ``GOPEX_AUDIO_VAE``, then a sibling ``ltx-2.5-audio-vae-*.safetensors``.
    """
    raw = os.environ.get("GOPEX_AUDIO_VAE", "").strip()
    if raw:
        env_path = Path(raw).expanduser()
        if env_path.is_file():
            return env_path.resolve()

    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_file() and "audio-vae" in path.name.lower():
        return path
    if path.is_file() and "audio_vae" in path.name.lower():
        return path

    search_dirs: list[Path] = []
    if path.is_file():
        search_dirs.append(path.parent)
    elif path.is_dir():
        search_dirs.append(path)
        search_dirs.append(path / "vae")
    env_video = _vae_path_from_env()
    if env_video is not None:
        search_dirs.append(env_video.parent)
    home25 = Path(os.environ.get("GOPEX_LTX25_HOME", Path.home() / "gopex-ltx-2.5")).expanduser()
    search_dirs.extend([home25 / "vae", home25 / "_hf_cache" / "vae"])

    names = (
        "ltx-2.5-audio-vae-bf16.safetensors",
        "ltx-2.5-audio-vae.safetensors",
    )
    seen: set[Path] = set()
    for directory in search_dirs:
        if directory in seen or not directory.is_dir():
            continue
        seen.add(directory)
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()

    # Unified 2.3-style checkpoints keep audio VAE in the same file as video.
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"Cannot resolve audio VAE from {checkpoint_path}; set GOPEX_AUDIO_VAE"
    )


def resolve_vae_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Return the base LTX checkpoint that carries VAE weights for a native or merged tree.

    For LTX-2.5 split packs, prefer ``GOPEX_PREP_VAE_CKPT`` / ``GOPEX_VIDEO_VAE`` or
    ``native_manifest.json`` ``vae_checkpoint`` pointing at the standalone DiffVAE/ConvVAE file.
    """
    env_vae = _vae_path_from_env()
    if env_vae is not None:
        return env_vae.resolve()

    path = Path(checkpoint_path).expanduser().resolve()
    if path.is_file():
        return path

    manifest = _read_native_manifest(path) if path.is_dir() else None
    if manifest:
        for key in ("vae_checkpoint", "source_ltx_ckpt"):
            raw = manifest.get(key)
            if raw:
                candidate = Path(str(raw)).expanduser().resolve()
                if candidate.is_file():
                    return candidate
        env_ckpt = _ltx_ckpt_from_env()
        if env_ckpt is not None:
            return env_ckpt.resolve()
        raise FileNotFoundError(
            f"Native fold at {path} has no vae_checkpoint in native_manifest.json and "
            "GOPEX_PREP_VAE_CKPT / LTX_CKPT are unset."
        )

    raise FileNotFoundError(f"Cannot resolve VAE checkpoint from {checkpoint_path}")


def peek_video_aggregate_embed_in_features(checkpoint_path: str | Path) -> int | None:
    """Read ``video_aggregate_embed.weight`` in_features from nearby safetensors without a full load."""
    try:
        import safetensors
    except ImportError:
        return None

    path = Path(checkpoint_path).expanduser().resolve()
    shard_paths: list[str] = []
    if path.is_file():
        shard_paths = sorted(
            {
                str(path),
                *[str(item) for item in path.parent.glob("*.safetensors") if item.is_file()],
                *[str(item) for item in path.parent.rglob("*.safetensors") if item.is_file()],
            }
        )
    elif path.is_dir():
        shard_paths = sorted(
            {str(item) for item in path.glob("*.safetensors") if item.is_file()}
            | {str(item) for item in path.rglob("*.safetensors") if item.is_file()}
        )
    else:
        return None

    flat_dims: list[int] = []
    for shard_path in sorted(dict.fromkeys(shard_paths))[:256]:
        try:
            with safetensors.safe_open(shard_path, framework="pt") as handle:
                for key in handle.keys():
                    if "video_aggregate_embed" in key and key.endswith(".weight"):
                        shape = handle.get_slice(key).get_shape()
                        if len(shape) == 2:
                            flat_dims.append(int(shape[1]))
        except (OSError, ValueError, RuntimeError, KeyError):
            continue
    return max(flat_dims) if flat_dims else None


def create_meta_model(
    configurator: type[ModelConfigurator[_M]],
    config: dict,
    module_ops: tuple[ModuleOps, ...] = (),
) -> _M:
    """Create a model on the meta device and apply module operations."""
    with torch.device("meta"):
        model = configurator.from_config(config)
    for op in module_ops:
        if op.matcher(model):
            model = op.mutator(model)
    return model
