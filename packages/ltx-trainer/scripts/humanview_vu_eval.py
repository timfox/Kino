#!/usr/bin/env python3
"""Human-View video MLLM survey CLI (Meng et al., arXiv:2606.07433)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.humanview_vu.benchmarks import benchmarks_bundle  # noqa: E402
from ltx_trainer.humanview_vu.datasets import datasets_card  # noqa: E402
from ltx_trainer.humanview_vu.future import future_card  # noqa: E402
from ltx_trainer.humanview_vu.paper import evaluation_demo, framework_card, knowledge_blob  # noqa: E402
from ltx_trainer.humanview_vu.pipeline import evaluation_demo_run  # noqa: E402
from ltx_trainer.humanview_vu.subfields import subfields_card  # noqa: E402
from ltx_trainer.humanview_vu.taxonomy import classify_method, taxonomy_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_blob(), indent=2))
    return 0


def _cmd_framework(_: argparse.Namespace) -> int:
    print(json.dumps(framework_card(), indent=2))
    return 0


def _cmd_taxonomy(_: argparse.Namespace) -> int:
    print(json.dumps(taxonomy_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_datasets(_: argparse.Namespace) -> int:
    print(json.dumps(datasets_card(), indent=2))
    return 0


def _cmd_subfields(_: argparse.Namespace) -> int:
    print(json.dumps(subfields_card(), indent=2))
    return 0


def _cmd_future(_: argparse.Namespace) -> int:
    print(json.dumps(future_card(), indent=2))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    print(json.dumps(classify_method(args.name), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo_run(num_frames=args.frames, query=args.query), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.humanview_vu.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_full_demo(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("framework", _cmd_framework),
        ("taxonomy", _cmd_taxonomy),
        ("tables", _cmd_tables),
        ("datasets", _cmd_datasets),
        ("subfields", _cmd_subfields),
        ("future", _cmd_future),
        ("smoke", _cmd_smoke),
        ("full-demo", _cmd_full_demo),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    p_classify = sub.add_parser("classify")
    p_classify.add_argument("name")
    p_classify.set_defaults(func=_cmd_classify)

    p_demo = sub.add_parser("demo")
    p_demo.add_argument("--frames", type=int, default=16)
    p_demo.add_argument("--query", default=None)
    p_demo.set_defaults(func=_cmd_demo)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
