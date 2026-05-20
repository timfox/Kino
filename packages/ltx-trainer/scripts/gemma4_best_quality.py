#!/usr/bin/env python3
# ruff: noqa: T201
"""Best-quality Gemma 4 + LTX 2.3 workflow (rank-512 bridge → fold → native LoRA).

Prints commands or runs steps when given ``--run``. See ``configs/GEMMA4_BEST_QUALITY.md``.

Example::

    python scripts/gemma4_best_quality.py plan
    python scripts/gemma4_best_quality.py probe
    python scripts/gemma4_best_quality.py preprocess --run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_PKG = _SCRIPTS.parent
_REPO = _PKG.parent.parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    import _kino_bootstrap  # noqa: F401
except ImportError:
    pass

# --- edit these once for your machine ---
DEFAULT_DATASET = Path(
    "/run/media/tim/Datasets/datasets/prelinger-archives-open/data/dataset.json"
)
DEFAULT_PRECOMPUTED = Path(
    "/run/media/tim/Datasets/datasets/prelinger-archives-open/.precomputed"
)
DEFAULT_LTX = Path("/home/tim/ComfyUI/models/checkpoints/ltx-2.3-22b-dev.safetensors")
DEFAULT_GEMMA = Path(
    "/home/tim/.cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/"
    "462a98a12e28e2cbcfccaf78fe41e3e50235e6ae"
)
DEFAULT_NATIVE_LTX = Path(
    "/run/media/tim/Datasets/datasets/prelinger-archives-open/ltx_runs/ltx-2.3-22b-gemma4-native.safetensors"
)
BRIDGE_RANK = 512
BUCKETS = "1024x576x41"


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print("$", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.call(cmd, cwd=cwd or _PKG, env=merged)


def cmd_plan(_: argparse.Namespace) -> int:
    print(
        """
Gemma 4 + LTX — best quality ladder (highest practical bridge rank 512)

  A. probe          — must understand mismatch (exit 2 → bridge path)
  B. preprocess     — latents + conditions @ flat_dim_bridge_rank 512 (NOT dense bridge)
  C. audit          — strict paired latents/conditions
  D. phase1a        — train text stack (configs/ltx2_text_stack_gemma4_rank512.yaml)
  E. fold           — bake bridge → native LTX .safetensors
  F. probe-native   — must exit 0
  G. preprocess-native — wipe conditions/, re-embed with native checkpoint (no bridge rank)
  H. phase2         — configs/ltx2_av_lora_gemma4_native_best.yaml (LoRA 128 + connectors)

Interim (skip fold): configs/ltx2_av_lora_gemma4_rank512_connectors.yaml after step B.

GPU: stop vLLM / other jobs before D and B caption phase. Caption embed uses rank 512 on GPU (~few GB, not 30GB dense).
"""
    )
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    probe = _REPO / "tools" / "ltx_gemma_flat_probe.py"
    cmd = [_py(), str(probe), "--ltx", str(args.ltx), "--gemma", str(args.gemma)]
    if args.run:
        return _run(cmd, cwd=_REPO)
    print(" ".join(cmd))
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    cmd = [
        _py(),
        str(_SCRIPTS / "process_dataset.py"),
        str(args.dataset),
        "--resolution-buckets",
        BUCKETS,
        "--model-path",
        str(args.ltx),
        "--text-encoder-path",
        str(args.gemma),
        "--output-dir",
        str(args.precomputed),
        "--flat-dim-bridge-rank",
        str(BRIDGE_RANK),
        "--skip-existing",
        "--device",
        "cuda",
        "--vae-tiling",
    ]
    if args.fresh_conditions:
        print("NOTE: --fresh-conditions requested: rm conditions/ manually before --run")
    if args.latents_only:
        cmd.append("--latents-only")
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    audit = _REPO / "tools" / "ltx_precomputed_audit.py"
    yaml = _PKG / "configs" / "ltx2_av_lora_gemma4_rank512_connectors.yaml"
    cmd = [
        _py(),
        str(audit),
        "--preprocessed-root",
        str(args.precomputed),
        "--training-yaml",
        str(yaml),
        "--strict",
    ]
    env = {"PYTHONPATH": str(_PKG / "src")}
    if args.run:
        return _run(cmd, cwd=_REPO, env=env)
    print("PYTHONPATH=kino/packages/ltx-trainer/src", " ".join(cmd[1:]))
    return 0


def cmd_phase1a(args: argparse.Namespace) -> int:
    cfg = _PKG / "configs" / "ltx2_text_stack_gemma4_rank512.yaml"
    cmd = [_py(), str(_SCRIPTS / "train.py"), str(cfg)]
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def cmd_phase0_interim(args: argparse.Namespace) -> int:
    cfg = _PKG / "configs" / "ltx2_av_lora_gemma4_rank512_connectors.yaml"
    cmd = [_py(), str(_SCRIPTS / "train.py"), str(cfg)]
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def cmd_phase2_native(args: argparse.Namespace) -> int:
    cfg = _PKG / "configs" / "ltx2_av_lora_gemma4_native_best.yaml"
    cmd = [_py(), str(_SCRIPTS / "train.py"), str(cfg)]
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def cmd_fold(args: argparse.Namespace) -> int:
    if not args.text_stack:
        print("Pass --text-stack path/to/text_stack_weights_step_XXXXX.safetensors", file=sys.stderr)
        return 1
    cmd = [
        _py(),
        str(_SCRIPTS / "fold_flat_dim_bridge.py"),
        "--ltx",
        str(args.ltx),
        "--gemma",
        str(args.gemma),
        "--text-stack",
        str(args.text_stack),
        "--output",
        str(args.native_ltx),
        "--flat-dim-bridge-rank",
        str(BRIDGE_RANK),
    ]
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def cmd_preprocess_native(args: argparse.Namespace) -> int:
    cmd = [
        _py(),
        str(_SCRIPTS / "process_dataset.py"),
        str(args.dataset),
        "--resolution-buckets",
        BUCKETS,
        "--model-path",
        str(args.native_ltx),
        "--text-encoder-path",
        str(args.gemma),
        "--output-dir",
        str(args.precomputed),
        "--skip-existing",
        "--device",
        "cuda",
        "--vae-tiling",
    ]
    print(
        "Before --run: remove conditions/ (keep latents/) so captions re-embed with native geometry.\n"
        f"  rm -rf {args.precomputed / 'conditions'}"
    )
    if args.run:
        return _run(cmd)
    print(" ".join(cmd))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    plan_sp = sub.add_parser("plan", help="Show the full quality ladder")
    plan_sp.set_defaults(func=cmd_plan)

    for name, fn in (
        ("probe", cmd_probe),
        ("preprocess", cmd_preprocess),
        ("audit", cmd_audit),
        ("phase1a", cmd_phase1a),
        ("phase0-interim", cmd_phase0_interim),
        ("phase2-native", cmd_phase2_native),
        ("fold", cmd_fold),
        ("preprocess-native", cmd_preprocess_native),
    ):
        sp = sub.add_parser(name.replace("-", "_") if False else name)
        sp.add_argument("--run", action="store_true", help="Execute command")
        sp.set_defaults(func=fn)

    # shared defaults on all subcommands except plan
    for name, sp in sub.choices.items():
        if name == "plan":
            continue
        sp.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        sp.add_argument("--precomputed", type=Path, default=DEFAULT_PRECOMPUTED)
        sp.add_argument("--ltx", type=Path, default=DEFAULT_LTX)
        sp.add_argument("--gemma", type=Path, default=DEFAULT_GEMMA)
        sp.add_argument("--native-ltx", type=Path, default=DEFAULT_NATIVE_LTX)

    preprocess_sp = sub.choices["preprocess"]
    preprocess_sp.add_argument("--latents-only", action="store_true")
    preprocess_sp.add_argument(
        "--fresh-conditions",
        action="store_true",
        help="Remind to delete conditions/ before full re-embed at rank 512",
    )

    fold_sp = sub.choices["fold"]
    fold_sp.add_argument("--text-stack", type=Path, default=None)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
