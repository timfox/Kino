"""CLI for AG-REPA stub."""

from __future__ import annotations

import argparse
import json

from ltx_trainer.ag_repa.config import AgRepaConfig
from ltx_trainer.ag_repa.mock import evaluation_smoke
from ltx_trainer.ag_repa.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AG-REPA causal layer REPA stub (arXiv:2603.01006)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("framework", help="Print framework card JSON")
    sub.add_parser("knowledge", help="Print knowledge card JSON")
    sub.add_parser("tables", help="Print paper tables bundle")
    sub.add_parser("smoke", help="Run evaluation smoke")
    demo = sub.add_parser("demo", help="Run evaluation demo")
    demo.add_argument("--strategy", default="ag_repa")

    args = p.parse_args(argv)
    cfg = AgRepaConfig()

    if args.cmd == "framework":
        print(json.dumps(framework_card(cfg), indent=2))
    elif args.cmd == "knowledge":
        print(json.dumps(knowledge_card(cfg), indent=2))
    elif args.cmd == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.cmd == "smoke":
        print(json.dumps(evaluation_smoke(cfg), indent=2))
    elif args.cmd == "demo":
        print(json.dumps(evaluation_demo(strategy=args.strategy, cfg=cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
