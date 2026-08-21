"""CLI: MER-with-LLMs survey framework, benchmarks, demo."""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_framework(_: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm import framework_card

    print(json.dumps(framework_card(), indent=2, ensure_ascii=False))
    return 0


def _cmd_benchmarks(_: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm import benchmarks_bundle

    print(json.dumps(benchmarks_bundle(), indent=2, ensure_ascii=False))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm import evaluation_demo

    print(json.dumps(evaluation_demo(), indent=2, ensure_ascii=False))
    return 0


def _cmd_knowledge(_: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm.knowledge import mer_llm_knowledge_blob

    print(json.dumps(mer_llm_knowledge_blob(), indent=2, ensure_ascii=False))
    return 0


def _cmd_radar(args: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm.catalog import research_radar_brief

    print(json.dumps(research_radar_brief(vertical=args.vertical), indent=2, ensure_ascii=False))
    return 0


def _cmd_taxonomy(_: argparse.Namespace) -> int:
    from ltx_trainer.mer_llm import five_subtasks, taxonomy_branches, three_challenges

    print(
        json.dumps(
            {
                "challenges": three_challenges(),
                "branches": taxonomy_branches(),
                "subtasks": five_subtasks(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MER-with-LLMs survey utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in (
        ("framework", _cmd_framework),
        ("benchmarks", _cmd_benchmarks),
        ("demo", _cmd_demo),
        ("taxonomy", _cmd_taxonomy),
        ("knowledge", _cmd_knowledge),
    ):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)

    p_r = sub.add_parser("radar", help="Research Radar brief for a vertical")
    p_r.add_argument(
        "vertical",
        nargs="?",
        default="game_live_ops",
        choices=["game_live_ops", "trailer_marketing", "political_speech"],
    )
    p_r.set_defaults(func=_cmd_radar)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
