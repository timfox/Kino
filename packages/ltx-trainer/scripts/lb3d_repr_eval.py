#!/usr/bin/env python3
"""LB3D repr survey CLI (Schockaert et al.; arXiv:2606.04871)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltx_trainer.lb3d_repr.config import PAPER_URL  # noqa: E402
from ltx_trainer.lb3d_repr.integration import (  # noqa: E402
    gopex_stub_links,
    ltx_nvs_pipeline_notes,
    survey_to_gopex_mapping,
)
from ltx_trainer.lb3d_repr.pipeline import (  # noqa: E402
    benchmarks_bundle,
    evaluation_demo,
    evaluation_smoke,
    framework_card,
)


def _cmd_links(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "stubs": gopex_stub_links(),
                "mapping": survey_to_gopex_mapping(),
                "ltx_notes": ltx_nvs_pipeline_notes(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_knowledge(_args: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_tables(_args: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(seed=_args.seed), indent=2))
    return 0


def _cmd_demo(_args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(seed=_args.seed), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=f"LB3D representations survey — {PAPER_URL}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("tables").set_defaults(func=_cmd_tables)
    sub.add_parser("links").set_defaults(func=_cmd_links)
    sm = sub.add_parser("smoke")
    sm.add_argument("--seed", type=int, default=0)
    sm.set_defaults(func=_cmd_smoke)
    dm = sub.add_parser("demo")
    dm.add_argument("--seed", type=int, default=0)
    dm.set_defaults(func=_cmd_demo)
    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
