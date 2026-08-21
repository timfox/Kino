#!/usr/bin/env python3
"""HoliTok holistic speech tokenizer CLI (arXiv:2605.29948)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.holitok.config import HoliTokConfig  # noqa: E402
from ltx_trainer.holitok.pipeline import benchmarks_bundle, evaluation_demo, framework_card, pipeline_demo  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="HoliTok continuous holistic tokenizer stub")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("knowledge")
    sp.set_defaults(func=lambda _: print(json.dumps(framework_card(), indent=2)) or 0)

    sp = sub.add_parser("tables")
    sp.set_defaults(func=lambda _: print(json.dumps(benchmarks_bundle(), indent=2)) or 0)

    sp = sub.add_parser("smoke")
    sp.set_defaults(
        func=lambda _: print(json.dumps(__import__("ltx_trainer.holitok.mock", fromlist=["evaluation_smoke"]).evaluation_smoke(), indent=2, default=str)) or 0
    )

    sp = sub.add_parser("demo")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=lambda a: print(json.dumps(pipeline_demo(HoliTokConfig(), seed=a.seed), indent=2)) or 0)

    sp = sub.add_parser("eval")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=lambda a: print(json.dumps(evaluation_demo(seed=a.seed), indent=2)) or 0)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
