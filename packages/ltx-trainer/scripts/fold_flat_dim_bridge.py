#!/usr/bin/env python3
"""Bake a trained flat_dim bridge into native ``video_aggregate_embed`` / ``audio_aggregate_embed`` weights.

After Phase 0/1 text-stack training, run::

    python scripts/fold_flat_dim_bridge.py \\
      --ltx /path/to/ltx-2.3-22b-dev.safetensors \\
      --gemma /path/to/gemma-4-snapshot \\
      --text-stack /path/to/checkpoints/text_stack_weights_step_01000.safetensors \\
      --output /path/to/ltx-2.3-22b-gemma4-native.safetensors

Then verify::

    python tools/ltx_gemma_flat_probe.py --ltx /path/to/ltx-2.3-22b-gemma4-native.safetensors --gemma /path/to/gemma-4-snapshot
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Allow running from scripts/ with kino src on path via bootstrap
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    import _kino_bootstrap  # noqa: F401
except ImportError:
    pass

_TRAINER_SRC = _SCRIPTS.parent / "src"
if str(_TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAINER_SRC))

from ltx_trainer.text_stack_utils import (  # noqa: E402
    apply_low_rank_bridge_to_aggregate,
    fold_metadata_summary,
    gemma_flat_dim_from_path,
    ltx_checkpoint_flat_dim,
)

try:
    from ltx_core.loader.helpers import read_model_config  # noqa: E402
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
except ImportError:
    read_model_config = None  # type: ignore[misc, assignment]
    SafetensorsModelStateDictLoader = None  # type: ignore[misc, assignment]

_BRIDGE_PREFIXES = (
    "flat_dim_bridge",
    "feature_extractor.flat_dim_bridge",
    "ltx_experimental_flat_dim_bridge",
)
_VIDEO_WEIGHT_SUFFIXES = (
    "video_aggregate_embed.weight",
    "feature_extractor.video_aggregate_embed.weight",
)
_VIDEO_BIAS_SUFFIXES = (
    "video_aggregate_embed.bias",
    "feature_extractor.video_aggregate_embed.bias",
)
_AUDIO_WEIGHT_SUFFIXES = (
    "audio_aggregate_embed.weight",
    "feature_extractor.audio_aggregate_embed.weight",
)
_AUDIO_BIAS_SUFFIXES = (
    "audio_aggregate_embed.bias",
    "feature_extractor.audio_aggregate_embed.bias",
)


def _collect_ltx_shard_paths(ltx_path: Path) -> list[Path]:
    """Return only the checkpoint the user asked for (never every ``*.safetensors`` in ComfyUI)."""
    path = ltx_path.expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.safetensors"))
    raise FileNotFoundError(f"LTX path not found: {ltx_path}")


def _key_matches(key: str, suffixes: tuple[str, ...]) -> bool:
    return any(key.endswith(s) or key == s for s in suffixes)


def _should_skip_bridge_key(key: str) -> bool:
    return any(p in key for p in _BRIDGE_PREFIXES)


def _fold_from_text_stack(text_stack: Path, *, gemma_flat: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Fold using only ``text_stack_weights`` (~3 GB), not the full 22B LTX load."""
    sd = load_file(str(text_stack))
    v_w_key = next((k for k in sd if _key_matches(k, _VIDEO_WEIGHT_SUFFIXES)), None)
    if v_w_key is None:
        raise KeyError("text_stack missing video_aggregate_embed.weight")
    v_w_raw = sd[v_w_key]
    b0 = sd.get("feature_extractor.flat_dim_bridge.0.weight")
    b1 = sd.get("feature_extractor.flat_dim_bridge.1.weight")
    if b0 is None or b1 is None:
        raise KeyError("text_stack missing flat_dim_bridge.0/1 weights (rank-512 bridge)")
    if int(b0.shape[1]) != gemma_flat:
        raise ValueError(f"Bridge out dim {int(b0.shape[1])} != Gemma flat {gemma_flat}")
    # Two matmuls (~6 GiB peak), not dense bridge [188160, 327936] (~246 GiB).
    v_w = apply_low_rank_bridge_to_aggregate(v_w_raw, b1, b0)

    v_b_key = next((k for k in sd if _key_matches(k, _VIDEO_BIAS_SUFFIXES)), None)
    v_b = sd[v_b_key].detach() if v_b_key else None

    a_w = a_b = None
    a_w_key = next((k for k in sd if _key_matches(k, _AUDIO_WEIGHT_SUFFIXES)), None)
    if a_w_key is not None:
        a_w_raw = sd[a_w_key]
        a_w = apply_low_rank_bridge_to_aggregate(a_w_raw, b1, b0)
        a_b_key = next((k for k in sd if _key_matches(k, _AUDIO_BIAS_SUFFIXES)), None)
        a_b = sd[a_b_key].detach() if a_b_key else None

    del sd
    gc.collect()
    return v_w, v_b, a_w, a_b


def _replacement_for_key(
    key: str,
    *,
    video_w: torch.Tensor,
    video_b: torch.Tensor | None,
    audio_w: torch.Tensor | None,
    audio_b: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if _key_matches(key, _VIDEO_WEIGHT_SUFFIXES):
        return video_w if video_w.dtype == dtype else video_w.to(dtype=dtype)
    if video_b is not None and _key_matches(key, _VIDEO_BIAS_SUFFIXES):
        return video_b if video_b.dtype == dtype else video_b.to(dtype=dtype)
    if audio_w is not None and _key_matches(key, _AUDIO_WEIGHT_SUFFIXES):
        return audio_w if audio_w.dtype == dtype else audio_w.to(dtype=dtype)
    if audio_b is not None and _key_matches(key, _AUDIO_BIAS_SUFFIXES):
        return audio_b if audio_b.dtype == dtype else audio_b.to(dtype=dtype)
    return None


def _tensor_dtype_from_slice(f: safe_open, key: str) -> torch.dtype:
    """Dtype without loading the full tensor into RAM."""
    dt = f.get_slice(key).get_dtype()
    if dt == "BF16":
        return torch.bfloat16
    if dt == "F16":
        return torch.float16
    if dt == "F32":
        return torch.float32
    return torch.float32


def _resolve_output_dir(output: Path) -> Path:
    """Use a directory for sharded output (avoids a second 43+ GiB RAM merge pass)."""
    out = output.expanduser().resolve()
    if out.suffix == ".safetensors":
        return out.with_suffix("")
    return out


def _canonical_aggregate_keys(
    *,
    video_w: torch.Tensor,
    video_b: torch.Tensor | None,
    audio_w: torch.Tensor | None,
    audio_b: torch.Tensor | None,
    key_prefix: str,
) -> dict[str, torch.Tensor]:
    """Keys to inject when the base transformer has no ``*_aggregate_embed`` tensors (LTX-2.5)."""
    prefix = key_prefix.rstrip(".")
    if prefix:
        prefix = prefix + "."
    out: dict[str, torch.Tensor] = {
        f"{prefix}feature_extractor.video_aggregate_embed.weight": video_w,
    }
    if video_b is not None:
        out[f"{prefix}feature_extractor.video_aggregate_embed.bias"] = video_b
    if audio_w is not None:
        out[f"{prefix}feature_extractor.audio_aggregate_embed.weight"] = audio_w
    if audio_b is not None:
        out[f"{prefix}feature_extractor.audio_aggregate_embed.bias"] = audio_b
    return out


def _stream_write_native_checkpoint(
    ltx_paths: list[Path],
    output_dir: Path,
    *,
    video_w: torch.Tensor,
    video_b: torch.Tensor | None,
    audio_w: torch.Tensor | None,
    audio_b: torch.Tensor | None,
    max_buffer_gb: float,
    inject_aggregates: bool = False,
    aggregate_key_prefix: str = "model.diffusion_model",
) -> tuple[int, list[Path]]:
    """Copy LTX into ``output_dir`` as shard files; peak RAM ≈ ``max_buffer_gb`` (+ folded aggregates)."""
    max_bytes = max(512 * 1024 * 1024, int(max_buffer_gb * (1024**3)))
    buffer: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    part_paths: list[Path] = []
    n_keys = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_keys: set[str] = set()
    replaced_any_aggregate = False

    def _flush() -> None:
        nonlocal buffer, buf_bytes
        if not buffer:
            return
        part = output_dir / f"native_{len(part_paths):05d}.safetensors"
        save_file(buffer, str(part))
        part_paths.append(part)
        print(f"  wrote {part.name} ({len(buffer)} tensors)", flush=True)
        buffer.clear()
        buf_bytes = 0
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    def _put(key: str, tensor: torch.Tensor) -> None:
        nonlocal buf_bytes, n_keys
        if key in seen_keys:
            return
        seen_keys.add(key)
        nbytes = tensor.numel() * tensor.element_size()
        buffer[key] = tensor
        buf_bytes += nbytes
        n_keys += 1
        if buf_bytes >= max_bytes:
            _flush()

    for sp in ltx_paths:
        print(f"  reading {sp} …", flush=True)
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                if _should_skip_bridge_key(key):
                    continue
                dtype = _tensor_dtype_from_slice(f, key)
                repl = _replacement_for_key(
                    key,
                    video_w=video_w,
                    video_b=video_b,
                    audio_w=audio_w,
                    audio_b=audio_b,
                    dtype=dtype,
                )
                if repl is not None:
                    replaced_any_aggregate = True
                tensor = repl if repl is not None else f.get_tensor(key)
                _put(key, tensor)

    if inject_aggregates and not replaced_any_aggregate:
        print("  injecting folded aggregates (base transformer had none) …", flush=True)
        for key, tensor in _canonical_aggregate_keys(
            video_w=video_w,
            video_b=video_b,
            audio_w=audio_w,
            audio_b=audio_b,
            key_prefix=aggregate_key_prefix,
        ).items():
            _put(key, tensor if tensor.dtype == torch.bfloat16 else tensor.to(dtype=torch.bfloat16))
    elif inject_aggregates and replaced_any_aggregate:
        print("  aggregates already present in base; inject skipped", flush=True)

    _flush()

    if not part_paths:
        raise RuntimeError("No tensors written from LTX checkpoint")
    return n_keys, part_paths


def _fold_from_existing_native(source: Path) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Copy already-folded aggregates from an existing native fold (no bridge re-apply)."""
    paths = _collect_ltx_shard_paths(source)
    v_w = v_b = a_w = a_b = None
    for sp in paths:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                if _should_skip_bridge_key(key):
                    continue
                if v_w is None and _key_matches(key, _VIDEO_WEIGHT_SUFFIXES):
                    v_w = f.get_tensor(key)
                elif v_b is None and _key_matches(key, _VIDEO_BIAS_SUFFIXES):
                    v_b = f.get_tensor(key)
                elif a_w is None and _key_matches(key, _AUDIO_WEIGHT_SUFFIXES):
                    a_w = f.get_tensor(key)
                elif a_b is None and _key_matches(key, _AUDIO_BIAS_SUFFIXES):
                    a_b = f.get_tensor(key)
        if v_w is not None and a_w is not None:
            break
    if v_w is None:
        raise KeyError(f"No video_aggregate_embed.weight under {source}")
    return v_w, v_b, a_w, a_b


def _write_native_manifest(output_dir: Path, part_paths: list[Path], meta: dict[str, object]) -> None:
    manifest: dict[str, object] = {
        "format": "gopex_native_ltx_shards",
        "parts": [p.name for p in part_paths],
    }
    for key, value in meta.items():
        if isinstance(value, (dict, list)):
            manifest[key] = value
        else:
            manifest[key] = str(value)
    (output_dir / "native_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ltx", type=Path, required=True, help="Base LTX .safetensors (single file or shard dir)")
    ap.add_argument("--gemma", type=Path, required=True, help="Gemma 4 HF snapshot (for flat_dim)")
    ap.add_argument(
        "--text-stack",
        type=Path,
        default=None,
        help="text_stack_weights_step_*.safetensors from training (required unless --copy-aggregates-from)",
    )
    ap.add_argument(
        "--copy-aggregates-from",
        type=Path,
        default=None,
        help="Existing native fold dir/file whose Gemma-flat aggregates are copied (skip bridge).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path: directory, or *.safetensors (writes sibling directory without .safetensors)",
    )
    ap.add_argument("--flat-dim-bridge-rank", type=int, default=None, help="Must match training if low-rank bridge")
    ap.add_argument(
        "--max-buffer-gb",
        type=float,
        default=None,
        help="Max RAM per write chunk (default: GOPEX_FOLD_MAX_BUFFER_GB or 4)",
    )
    ap.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="Standalone video VAE path for native_manifest (LTX-2.5 ConvVAE/DiffVAE). "
        "Defaults to --ltx when that file embeds a VAE; otherwise GOPEX_PREP_VAE_CKPT.",
    )
    ap.add_argument(
        "--inject-aggregates",
        action="store_true",
        help="Inject folded video/audio aggregates when the base transformer has none (LTX-2.5).",
    )
    ap.add_argument(
        "--aggregate-key-prefix",
        type=str,
        default="model.diffusion_model",
        help="Prefix for injected aggregate keys (default: model.diffusion_model).",
    )
    ap.add_argument(
        "--official-flat-dim",
        type=int,
        default=None,
        help="Official LTX flat dim when --ltx has no video_aggregate_embed (default 188160).",
    )
    args = ap.parse_args()

    # Fold is CPU-only; avoid accidental GPU allocation from other jobs' CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.official_flat_dim is not None:
        os.environ["GOPEX_LTX_OFFICIAL_FLAT_DIM"] = str(args.official_flat_dim)
    elif args.inject_aggregates and not os.environ.get("GOPEX_LTX_OFFICIAL_FLAT_DIM"):
        os.environ["GOPEX_LTX_OFFICIAL_FLAT_DIM"] = "188160"

    max_buffer_gb = float(
        args.max_buffer_gb
        if args.max_buffer_gb is not None
        else os.environ.get("GOPEX_FOLD_MAX_BUFFER_GB", "2")
    )
    output_dir = _resolve_output_dir(args.output)

    if args.copy_aggregates_from is None and args.text_stack is None:
        print("Need --text-stack or --copy-aggregates-from", file=sys.stderr)
        return 2

    gemma_flat = gemma_flat_dim_from_path(args.gemma)
    try:
        ckpt_flat = ltx_checkpoint_flat_dim(args.ltx)
    except ValueError:
        if args.inject_aggregates or args.copy_aggregates_from or os.environ.get("GOPEX_LTX_OFFICIAL_FLAT_DIM"):
            ckpt_flat = int(os.environ.get("GOPEX_LTX_OFFICIAL_FLAT_DIM", "188160"))
            print(f"No aggregates in --ltx; using official flat dim {ckpt_flat}", flush=True)
        else:
            raise
    if args.copy_aggregates_from is not None:
        print(f"Copying folded aggregates from {args.copy_aggregates_from} …", flush=True)
        v_w, v_b, a_w, a_b = _fold_from_existing_native(args.copy_aggregates_from)
        if int(v_w.shape[1]) != gemma_flat:
            raise ValueError(
                f"Source aggregate in_features {int(v_w.shape[1])} != Gemma flat {gemma_flat}"
            )
        ckpt_flat = int(v_w.shape[1])  # already native
    else:
        if ckpt_flat == gemma_flat:
            print(f"Checkpoint already native ({gemma_flat}); copying without fold.")
        elif ckpt_flat < gemma_flat:
            if args.flat_dim_bridge_rank is None:
                print(
                    f"Checkpoint flat {ckpt_flat} < Gemma {gemma_flat}; "
                    "pass --flat-dim-bridge-rank (same as Phase 1a) to bake the trained bridge.",
                    file=sys.stderr,
                )
                return 2
            print(
                f"Expanding aggregates {ckpt_flat} → {gemma_flat} via rank-{args.flat_dim_bridge_rank} bridge.",
                flush=True,
            )

        print(f"Folding from text stack only ({args.text_stack.name}) …", flush=True)
        v_w, v_b, a_w, a_b = _fold_from_text_stack(args.text_stack, gemma_flat=gemma_flat)

    v_w = v_w.to(torch.bfloat16)
    v_b = v_b.to(torch.bfloat16) if v_b is not None else None
    if a_w is not None:
        a_w = a_w.to(torch.bfloat16)
        a_b = a_b.to(torch.bfloat16) if a_b is not None else None
    print(f"  video weight {tuple(v_w.shape)} bf16", flush=True)
    if a_w is not None:
        print(f"  audio weight {tuple(a_w.shape)} bf16", flush=True)
    gc.collect()

    shard_paths = _collect_ltx_shard_paths(args.ltx)
    total_gb = sum(p.stat().st_size for p in shard_paths) / (1024**3)
    print(f"LTX shard(s): {len(shard_paths)} file(s), ~{total_gb:.1f} GiB on disk", flush=True)
    if len(shard_paths) > 1:
        print("  (directory mode — all shards under that folder)", flush=True)
    elif shard_paths[0].parent.name == "checkpoints" and len(list(shard_paths[0].parent.glob("*.safetensors"))) > 3:
        print(
            "  NOTE: only the path you passed is used (not every file in ComfyUI/checkpoints).",
            flush=True,
        )

    print(
        f"Writing native shards under {output_dir} (≤{max_buffer_gb:.0f} GiB RAM per chunk, no merge pass) …",
        flush=True,
    )
    n_keys, part_paths = _stream_write_native_checkpoint(
        shard_paths,
        output_dir,
        video_w=v_w,
        video_b=v_b,
        audio_w=a_w,
        audio_b=a_b,
        max_buffer_gb=max_buffer_gb,
        inject_aggregates=bool(args.inject_aggregates or args.copy_aggregates_from),
        aggregate_key_prefix=str(args.aggregate_key_prefix),
    )

    meta = fold_metadata_summary(
        gemma_flat=gemma_flat,
        ckpt_flat_before=ckpt_flat,
        out_features=int(v_w.shape[0]),
    )
    source_ltx = str(shard_paths[0].resolve()) if len(shard_paths) == 1 else str(args.ltx.resolve())
    meta["source_ltx_ckpt"] = source_ltx
    vae_ckpt = args.vae_checkpoint
    if vae_ckpt is None:
        env_vae = os.environ.get("GOPEX_PREP_VAE_CKPT") or os.environ.get("GOPEX_VIDEO_VAE")
        vae_ckpt = Path(env_vae) if env_vae else None
    if vae_ckpt is not None and Path(vae_ckpt).expanduser().is_file():
        meta["vae_checkpoint"] = str(Path(vae_ckpt).expanduser().resolve())
    else:
        meta["vae_checkpoint"] = source_ltx
    meta["native_model_label"] = output_dir.name
    meta["product_version"] = (
        "2.5" if args.inject_aggregates or args.copy_aggregates_from or "2.5" in output_dir.name else "2.3"
    )
    if args.copy_aggregates_from is not None:
        meta["aggregates_copied_from"] = str(Path(args.copy_aggregates_from).expanduser().resolve())
    if read_model_config is not None and SafetensorsModelStateDictLoader is not None:
        ltx_cfg = read_model_config(source_ltx, SafetensorsModelStateDictLoader())
        if not ltx_cfg and args.copy_aggregates_from is not None:
            # Prefer schema from the prior Gopex native fold when 2.5 transformer has none.
            src_manifest = Path(args.copy_aggregates_from).expanduser()
            if src_manifest.is_dir():
                mf = src_manifest / "native_manifest.json"
                if mf.is_file():
                    try:
                        prior = json.loads(mf.read_text(encoding="utf-8"))
                        if isinstance(prior.get("ltx_config"), dict):
                            ltx_cfg = prior["ltx_config"]
                    except (OSError, json.JSONDecodeError):
                        pass
        if ltx_cfg:
            meta["ltx_config"] = ltx_cfg
        else:
            meta["ltx_config_note"] = "source transformer had no embedded config; use runtime defaults"
    _write_native_manifest(output_dir, part_paths, meta)
    print(f"Wrote {len(part_paths)} shard(s), {n_keys} tensors → {output_dir}", flush=True)
    print(f"  Use this directory as model_path / NATIVE_LTX (not a single .safetensors path).", flush=True)
    print(f"video_aggregate_embed: [{v_w.shape[0]}, {v_w.shape[1]}] (Gemma flat {gemma_flat})")
    print(json.dumps({k: v for k, v in meta.items() if k != "ltx_config"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
