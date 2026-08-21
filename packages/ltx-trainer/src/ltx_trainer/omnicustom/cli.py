"""CLI for OmniCustom stub."""

from __future__ import annotations

import argparse
import json

from ltx_trainer.omnicustom.config import OmniCustomConfig
from ltx_trainer.omnicustom.mock import evaluation_smoke
from ltx_trainer.omnicustom.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="OmniCustom sync AV customization stub (arXiv:2602.12304)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("framework", help="Print framework card JSON")
    sub.add_parser("knowledge", help="Print knowledge card JSON")
    sub.add_parser("tables", help="Print paper tables bundle")
    smoke = sub.add_parser("smoke", help="Run evaluation smoke")
    demo = sub.add_parser("demo", help="Run evaluation demo for a benchmark case")
    demo.add_argument("--case-id", default="sydney_harbour")

    args = p.parse_args(argv)
    cfg = OmniCustomConfig()

    if args.cmd == "framework":
        print(json.dumps(framework_card(cfg), indent=2))
    elif args.cmd == "knowledge":
        print(json.dumps(knowledge_card(cfg), indent=2))
    elif args.cmd == "tables":
        print(json.dumps(benchmarks_bundle(), indent=2))
    elif args.cmd == "smoke":
        print(json.dumps(evaluation_smoke(cfg), indent=2))
    elif args.cmd == "demo":
        print(json.dumps(evaluation_demo(case_id=args.case_id, cfg=cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
