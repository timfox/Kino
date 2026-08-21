#!/usr/bin/env python3
"""Trajectory-guided maritime I2V reconstruction CLI (Bompai et al. arXiv:2605.16420)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.traj_i2v.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.traj_i2v.config import PAPER_URL, TrajI2VConfig  # noqa: E402
from ltx_trainer.traj_i2v.paper import framework_card  # noqa: E402
from ltx_trainer.traj_i2v.pipeline import evaluation_demo_run, reconstruct_clip, save_conditioning_json  # noqa: E402
from ltx_trainer.traj_i2v.synthetic import (  # noqa: E402
    synthetic_anchors,
    synthetic_gps_log,
    synthetic_reference_frame,
)


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.traj_i2v.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo_run(device=args.device)
    print(json.dumps(out, indent=2))
    return 0


def _cmd_synthetic(args: argparse.Namespace) -> int:
    cfg = TrajI2VConfig(
        image_width=args.width,
        image_height=args.height,
        num_frames=args.frames,
    )
    ref = synthetic_reference_frame(cfg)
    log = synthetic_gps_log(cfg)
    anchors = synthetic_anchors(cfg)
    result = reconstruct_clip(ref, log, anchors, cfg=cfg, method=args.method)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_conditioning_json(result["conditioning"], out_dir / "sg_i2v_conditioning.json")
    meta = {
        "paper": PAPER_URL,
        "method": args.method,
        "metrics": result["metrics"],
        "scale_px_per_m": result["scale_px_per_m"],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Traj-I2V maritime reconstruction (arXiv:2605.16420)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    d = sub.add_parser("demo", help="Synthetic pipeline demo")
    d.add_argument("--device", default="cpu")
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("synthetic", help="Run synthetic clip reconstruction")
    s.add_argument("-o", "--output-dir", default="traj_i2v_out")
    s.add_argument("--method", choices=("sg_i2v", "optical_flow", "rife"), default="sg_i2v")
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--height", type=int, default=144)
    s.add_argument("--frames", type=int, default=14)
    s.set_defaults(func=_cmd_synthetic)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
