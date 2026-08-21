#!/usr/bin/env python3
"""EIGENET few-shot novel-view RIR CLI (arXiv:2605.28101)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.eigenet.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.eigenet.config import EigeNetConfig  # noqa: E402
from ltx_trainer.eigenet.pipeline import evaluation_demo, evaluation_smoke, framework_card, knowledge_card  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="EIGENET geometry-informed RIR stub")
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [
        ("knowledge", lambda _: print(json.dumps(knowledge_card(), indent=2)) or 0),
        ("framework", lambda _: print(json.dumps(framework_card(), indent=2)) or 0),
        ("tables", lambda _: print(json.dumps(benchmarks_bundle(), indent=2)) or 0),
        ("smoke", lambda _: print(json.dumps(evaluation_smoke(), indent=2)) or 0),
        ("demo", lambda _: print(json.dumps(evaluation_demo(EigeNetConfig()), indent=2)) or 0),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
