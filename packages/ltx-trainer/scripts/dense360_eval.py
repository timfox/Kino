#!/usr/bin/env python3
"""Dense360 CLI (Zhou et al., arXiv:2506.14471)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.dense360.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.dense360.config import PAPER_URL, Dense360Config  # noqa: E402
from ltx_trainer.dense360.datasets import datasets_card  # noqa: E402
from ltx_trainer.dense360.erp_rope import erp_position_grid, horizontal_w_index  # noqa: E402
from ltx_trainer.dense360.paper import framework_card  # noqa: E402
from ltx_trainer.dense360.pipeline import (  # noqa: E402
    ablation_erp_rope,
    evaluation_demo_run,
    slicing_demo,
    train_step,
)


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
    from ltx_trainer.dense360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_erp_rope(args: argparse.Namespace) -> int:
    w = horizontal_w_index(args.width).tolist()
    row_h, col_w = erp_position_grid(args.height, args.width)
    print(
        json.dumps(
            {
                "f_w_head": w[:8],
                "f_w_tail": w[-8:],
                "center_f": float(col_w[args.height // 2, args.width // 2]),
                "pole_f": float(col_w[0, args.width // 2]),
            },
            indent=2,
        )
    )
    return 0


def _cmd_ablation(args: argparse.Namespace) -> int:
    from ltx_trainer.dense360.benchmarks import TABLE4_ABLATION

    cfg = Dense360Config(height=args.height, width=args.width)
    print(
        json.dumps(
            {"table4_paper": TABLE4_ABLATION, "stub": ablation_erp_rope(cfg)},
            indent=2,
        )
    )
    return 0


def _cmd_slicing(args: argparse.Namespace) -> int:
    cfg = Dense360Config(height=args.height, width=args.width)
    print(json.dumps(slicing_demo(cfg), indent=2))
    return 0


def _cmd_train_stub(args: argparse.Namespace) -> int:
    cfg = Dense360Config(height=args.height, width=args.width)
    print(json.dumps(train_step(cfg), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = Dense360Config(height=args.height, width=args.width)
    print(json.dumps(evaluation_demo_run(cfg), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"Dense360 — {PAPER_URL}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("dataset").set_defaults(func=_cmd_dataset)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)
    erp = sub.add_parser("erp-rope")
    erp.add_argument("--height", type=int, default=256)
    erp.add_argument("--width", type=int, default=512)
    erp.set_defaults(func=_cmd_erp_rope)
    ab = sub.add_parser("ablation")
    ab.add_argument("--height", type=int, default=128)
    ab.add_argument("--width", type=int, default=256)
    ab.set_defaults(func=_cmd_ablation)
    sl = sub.add_parser("slicing")
    sl.add_argument("--height", type=int, default=64)
    sl.add_argument("--width", type=int, default=256)
    sl.set_defaults(func=_cmd_slicing)
    tr = sub.add_parser("train-stub")
    tr.add_argument("--height", type=int, default=64)
    tr.add_argument("--width", type=int, default=128)
    tr.set_defaults(func=_cmd_train_stub)
    demo = sub.add_parser("demo")
    demo.add_argument("--height", type=int, default=128)
    demo.add_argument("--width", type=int, default=256)
    demo.set_defaults(func=_cmd_demo)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
