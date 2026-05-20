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
import json
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

from ltx_trainer.model_loader import load_embeddings_processor  # noqa: E402
from ltx_trainer.text_stack_utils import (  # noqa: E402
    fold_aggregate_with_bridge,
    fold_metadata_summary,
    gemma_flat_dim_from_path,
    load_feature_extractor_state_dict,
    ltx_checkpoint_flat_dim,
)


def _collect_ltx_shard_paths(ltx_path: Path) -> list[Path]:
    if ltx_path.is_file():
        return sorted({ltx_path, *ltx_path.parent.glob("*.safetensors")})
    return sorted(ltx_path.rglob("*.safetensors"))


def _merge_folded_aggregates(
    ltx_paths: list[Path],
    *,
    video_w: torch.Tensor,
    video_b: torch.Tensor | None,
    audio_w: torch.Tensor | None,
    audio_b: torch.Tensor | None,
    gemma_flat: int,
) -> dict[str, torch.Tensor]:
    """Load all keys from LTX shard(s), replacing aggregate embed weights."""
    merged: dict[str, torch.Tensor] = {}
    v_keys = ("video_aggregate_embed.weight", "feature_extractor.video_aggregate_embed.weight")
    a_keys = ("audio_aggregate_embed.weight", "feature_extractor.audio_aggregate_embed.weight")
    vb_keys = ("video_aggregate_embed.bias", "feature_extractor.video_aggregate_embed.bias")
    ab_keys = ("audio_aggregate_embed.bias", "feature_extractor.audio_aggregate_embed.bias")
    bridge_prefixes = (
        "flat_dim_bridge",
        "feature_extractor.flat_dim_bridge",
        "ltx_experimental_flat_dim_bridge",
    )

    for sp in ltx_paths:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.endswith(p) or p in key for p in bridge_prefixes):
                    continue
                tensor = f.get_tensor(key)
                replaced = False
                for vk in v_keys:
                    if key.endswith(vk) or key == vk:
                        merged[key] = video_w.to(dtype=tensor.dtype)
                        if video_b is not None:
                            bk = key.replace(".weight", ".bias")
                            merged[bk] = video_b.to(dtype=tensor.dtype)
                        replaced = True
                        break
                if replaced:
                    continue
                if audio_w is not None:
                    for ak in a_keys:
                        if key.endswith(ak) or key == ak:
                            merged[key] = audio_w.to(dtype=tensor.dtype)
                            if audio_b is not None:
                                bk = key.replace(".weight", ".bias")
                                merged[bk] = audio_b.to(dtype=tensor.dtype)
                            replaced = True
                            break
                if replaced:
                    continue
                if key not in merged:
                    merged[key] = tensor

    if not merged:
        raise RuntimeError("No tensors loaded from LTX checkpoint")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ltx", type=Path, required=True, help="Base LTX .safetensors (or directory of shards)")
    ap.add_argument("--gemma", type=Path, required=True, help="Gemma 4 HF snapshot (for flat_dim)")
    ap.add_argument(
        "--text-stack",
        type=Path,
        required=True,
        help="text_stack_weights_step_*.safetensors from training",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output native-geometry .safetensors path")
    ap.add_argument(
        "--text-encoder-path",
        type=Path,
        default=None,
        help="Optional; defaults to --gemma (for load_embeddings_processor)",
    )
    ap.add_argument("--flat-dim-bridge-rank", type=int, default=None, help="Must match training if low-rank bridge")
    args = ap.parse_args()

    gemma_flat = gemma_flat_dim_from_path(args.gemma)
    ckpt_flat = ltx_checkpoint_flat_dim(args.ltx)
    if ckpt_flat == gemma_flat:
        print(f"Checkpoint already native ({gemma_flat}); copying without fold.")
    elif ckpt_flat < gemma_flat:
        print(f"Checkpoint flat {ckpt_flat} < Gemma {gemma_flat}; cannot fold.", file=sys.stderr)
        return 2

    te_path = args.text_encoder_path or args.gemma
    proc = load_embeddings_processor(
        checkpoint_path=args.ltx,
        device="cpu",
        dtype=torch.bfloat16,
        gemma_model_path=te_path,
        require_matched_gemma_text_flat_dim=False,
        flat_dim_bridge_rank=args.flat_dim_bridge_rank,
    )
    load_feature_extractor_state_dict(proc.feature_extractor, load_file(args.text_stack))

    fe = proc.feature_extractor
    if fe is None:
        print("feature_extractor missing after load", file=sys.stderr)
        return 1

    v_lin = fe.video_aggregate_embed
    bridge = getattr(fe, "flat_dim_bridge", None)
    v_w = fold_aggregate_with_bridge(v_lin.weight.detach().float(), bridge, gemma_flat_in=gemma_flat)
    v_b = v_lin.bias.detach() if v_lin.bias is not None else None

    a_w = a_b = None
    if fe.audio_aggregate_embed is not None:
        a_lin = fe.audio_aggregate_embed
        a_w = fold_aggregate_with_bridge(a_lin.weight.detach().float(), bridge, gemma_flat_in=gemma_flat)
        a_b = a_lin.bias.detach() if a_lin.bias is not None else None

    out_state = _merge_folded_aggregates(
        _collect_ltx_shard_paths(args.ltx),
        video_w=v_w,
        video_b=v_b,
        audio_w=a_w,
        audio_b=a_b,
        gemma_flat=gemma_flat,
    )

    meta = fold_metadata_summary(
        gemma_flat=gemma_flat,
        ckpt_flat_before=ckpt_flat,
        out_features=int(v_w.shape[0]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_state, str(args.output), metadata={k: str(v) for k, v in meta.items()})
    print(f"Wrote {args.output} ({len(out_state)} tensors)")
    print(f"video_aggregate_embed: [{v_w.shape[0]}, {v_w.shape[1]}] (Gemma flat {gemma_flat})")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
