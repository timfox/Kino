#!/usr/bin/env python3
"""torch-webgpu / WebGPU dispatch overhead CLI (Maczan arXiv:2604.02344)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.torch_webgpu.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.torch_webgpu.config import PAPER_URL, TorchWebGPUConfig  # noqa: E402
from ltx_trainer.torch_webgpu.integration import integration_bundle  # noqa: E402
from ltx_trainer.torch_webgpu.paper import evaluation_demo, framework_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.torch_webgpu.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(integration_bundle(), indent=2))
    return 0


def _cmd_dispatch(args: argparse.Namespace) -> int:
    from ltx_trainer.torch_webgpu.dispatch import dispatch_demo

    print(json.dumps(dispatch_demo(seed=args.seed), indent=2))
    return 0


def _cmd_fusion(_: argparse.Namespace) -> int:
    from ltx_trainer.torch_webgpu.fusion import fusion_demo

    print(json.dumps(fusion_demo(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out = evaluation_demo(seed=args.seed)
    out["config"] = TorchWebGPUConfig(
        per_dispatch_us=args.per_dispatch_us,
    ).__dict__
    out["paper"] = PAPER_URL
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="torch-webgpu WebGPU dispatch overhead evaluation CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("smoke")
    sub.add_parser("plan")

    dispatch = sub.add_parser("dispatch")
    dispatch.add_argument("--seed", type=int, default=0)

    sub.add_parser("fusion")

    demo = sub.add_parser("demo")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--per-dispatch-us", type=float, default=23.8, dest="per_dispatch_us")

    args = p.parse_args()
    return {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "smoke": _cmd_smoke,
        "plan": _cmd_plan,
        "dispatch": _cmd_dispatch,
        "fusion": _cmd_fusion,
        "demo": _cmd_demo,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
