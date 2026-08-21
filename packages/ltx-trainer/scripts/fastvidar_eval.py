#!/usr/bin/env python3
"""FastViDAR CLI (Zhao et al., arXiv:2509.23733)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.fastvidar.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.fastvidar.complexity import theoretical_speedup  # noqa: E402
from ltx_trainer.fastvidar.config import PAPER_URL, FastViDARConfig  # noqa: E402
from ltx_trainer.fastvidar.datasets import datasets_card  # noqa: E402
from ltx_trainer.fastvidar.paper import framework_card  # noqa: E402
from ltx_trainer.fastvidar.pipeline import evaluation_demo_run, train_step  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.fastvidar.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_complexity(_: argparse.Namespace) -> int:
    print(json.dumps(theoretical_speedup(), indent=2))
    return 0


def _cmd_ablation(_: argparse.Namespace) -> int:
    from ltx_trainer.fastvidar.benchmarks import TABLE1_AHA
    from ltx_trainer.fastvidar.pipeline import ablation_global_attention

    cfg = FastViDARConfig(height=args.height, width=args.width)
    print(
        json.dumps(
            {"table1_paper": TABLE1_AHA, "stub_forward": ablation_global_attention(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_fusion(_: argparse.Namespace) -> int:
    from ltx_trainer.fastvidar.benchmarks import TABLE2_FUSION
    from ltx_trainer.fastvidar.pipeline import fusion_strategy_demo

    print(
        json.dumps(
            {"table2_paper": TABLE2_FUSION, "stub_forward": fusion_strategy_demo()},
            indent=2,
        )
    )
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = FastViDARConfig(height=args.height, width=args.width)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = FastViDARConfig(height=args.height, width=args.width)
    print(json.dumps({"paper": PAPER_URL, **evaluation_demo_run(cfg)}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="FastViDAR omnidirectional depth")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    sub.add_parser("complexity").set_defaults(func=_cmd_complexity)
    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=64)
    ab.add_argument("--width", type=int, default=128)
    ab.set_defaults(func=_cmd_ablation)
    sub.add_parser("fusion").set_defaults(func=_cmd_fusion)
    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=64)
    tr.add_argument("--width", type=int, default=128)
    tr.set_defaults(func=_cmd_train_stub)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=64)
    demo.add_argument("--width", type=int, default=128)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
