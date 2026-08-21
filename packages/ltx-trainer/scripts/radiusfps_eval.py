#!/usr/bin/env python3
"""RadiusFPS CLI (Yu et al., arXiv:2606.06255)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.radiusfps.benchmarks import (  # noqa: E402
    fig1_fps_share,
    fig15_peak_speedups,
    summary_anchors,
    table_2_pointmetabase,
    table_4_ablation,
)
from ltx_trainer.radiusfps.config import RadiusFpsConfig  # noqa: E402
from ltx_trainer.radiusfps.fps import fps_algorithm_card  # noqa: E402
from ltx_trainer.radiusfps.gpu import fusion_kernel_card, gpu_pipeline_stages  # noqa: E402
from ltx_trainer.radiusfps.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.radiusfps.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.radiusfps.pipeline import run_demo  # noqa: E402
from ltx_trainer.radiusfps.voxel import voxel_method_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "table_2": table_2_pointmetabase(),
                "table_4": table_4_ablation(),
                "fig_1": fig1_fps_share(),
                "fig_15": fig15_peak_speedups(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "fps": fps_algorithm_card(),
                "radiusfps": voxel_method_card(),
                "radiusfps_g": fusion_kernel_card(),
                "gpu_stages": gpu_pipeline_stages(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = RadiusFpsConfig(num_samples=args.samples, nvox=args.nvox)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RadiusFPS spherical voxel FPS stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--samples", type=int, default=32)
    demo.add_argument("--nvox", type=int, default=16)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
