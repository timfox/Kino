#!/usr/bin/env python3
"""PlanAudio compositional unified audio CLI (arXiv:2605.28063)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.planaudio.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.planaudio.config import PlanAudioConfig  # noqa: E402
from ltx_trainer.planaudio.pipeline import evaluation_demo, framework_card, knowledge_card  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="PlanAudio free-form text-to-unified-audio stub")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=lambda _: print(json.dumps(framework_card(), indent=2)) or 0)

    sp = sub.add_parser("tables")
    sp.set_defaults(func=lambda _: print(json.dumps(benchmarks_bundle(), indent=2)) or 0)

    sp = sub.add_parser("smoke")
    sp.set_defaults(
        func=lambda _: print(json.dumps(__import__("ltx_trainer.planaudio.mock", fromlist=["evaluation_smoke"]).evaluation_smoke(), indent=2)) or 0
    )

    sp = sub.add_parser("demo")
    sp.set_defaults(func=lambda _: print(json.dumps(evaluation_demo(PlanAudioConfig()), indent=2)) or 0)

    sp = sub.add_parser("eval")
    sp.set_defaults(func=lambda _: print(json.dumps(evaluation_demo(PlanAudioConfig()), indent=2)) or 0)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
