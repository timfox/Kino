#!/usr/bin/env python3
"""Sphere360 Hub bridge + HDR timelapse LTX plan CLI (omniaudio/Sphere360)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.sphere360.benchmarks import PAPER_HUB, benchmarks_bundle  # noqa: E402
from ltx_trainer.sphere360.paper import framework_card  # noqa: E402
from ltx_trainer.sphere360.pipeline import evaluation_demo, pipeline_demo  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    from ltx_trainer.sphere360._bootstrap import ensure_repo_root

    ensure_repo_root()
    from gopex_datasets.sphere360.knowledge import sphere360_knowledge_blob

    kb = sphere360_knowledge_blob()
    print(
        json.dumps(
            {
                "hub": PAPER_HUB,
                "framework": framework_card(),
                "scale": kb["scale"],
                "license": kb.get("license"),
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from ltx_trainer.sphere360._bootstrap import ensure_repo_root

    ensure_repo_root()
    from gopex_datasets.sphere360.ltx_bridge import hdr_timelapse_lora_plan

    plan = hdr_timelapse_lora_plan(project_name=args.project)
    print(json.dumps(plan, indent=2))
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from ltx_trainer.sphere360._bootstrap import ensure_repo_root

    ensure_repo_root()
    from gopex_datasets.sphere360 import parse_clip_id

    ref = parse_clip_id(args.clip_id)
    print(
        json.dumps(
            {
                "clip_id": ref.clip_id,
                "youtube_video_id": ref.youtube_video_id,
                "segment_index": ref.segment_index,
            },
            indent=2,
        )
    )
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.sphere360.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(pipeline_demo(project=args.project), indent=2))
    return 0


def _cmd_eval(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Sphere360 HDR timelapse LTX bridge")
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("smoke", _cmd_smoke),
        ("eval", _cmd_eval),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)

    sp = sub.add_parser("plan")
    sp.add_argument("--project", default="sphere360-hdr-timelapse")
    sp.set_defaults(func=_cmd_plan)

    sp = sub.add_parser("parse")
    sp.add_argument("clip_id")
    sp.set_defaults(func=_cmd_parse)

    sp = sub.add_parser("demo")
    sp.add_argument("--project", default="sphere360-hdr-timelapse")
    sp.set_defaults(func=_cmd_demo)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
