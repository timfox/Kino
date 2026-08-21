"""CLI for VeriTrip benchmark utilities (arXiv:2605.28683)."""

from __future__ import annotations

import argparse
import json
import sys

from ltx_trainer.veritrip.pipeline import benchmarks_bundle, evaluation_demo, framework_card, knowledge_card


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VeriTrip travel planning benchmark utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("framework", help="Framework card (JSON)")
    sub.add_parser("knowledge", help="Compact knowledge card")
    sub.add_parser("benchmarks", help="Paper table excerpts")
    sub.add_parser("demo", help="MRB retrieval + VKB evaluation smoke")

    args = p.parse_args(argv)
    if args.cmd == "framework":
        _print(framework_card())
    elif args.cmd == "knowledge":
        _print(knowledge_card())
    elif args.cmd == "benchmarks":
        _print(benchmarks_bundle())
    elif args.cmd == "demo":
        _print(evaluation_demo())
    return 0


if __name__ == "__main__":
    sys.exit(main())
