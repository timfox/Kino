#!/usr/bin/env python3
"""AI-driven HPC workflows CLI (Alnasir, arXiv:2606.07491)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.ai_hpc_workflows.config import AiHpcWorkflowConfig  # noqa: E402
from ltx_trainer.ai_hpc_workflows.data_gravity import (  # noqa: E402
    data_gravity_assessment,
    io_mitigation_plan,
)
from ltx_trainer.ai_hpc_workflows.feedback import feedback_loop_spec, iteration_gate  # noqa: E402
from ltx_trainer.ai_hpc_workflows.orchestration import (  # noqa: E402
    architecture_card,
    distributed_training_advice,
    job_array_plan,
    recommend_orchestrator,
)
from ltx_trainer.ai_hpc_workflows.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.ai_hpc_workflows.phases import default_phase_plan, map_task_to_resource  # noqa: E402
from ltx_trainer.ai_hpc_workflows.pipeline import run_demo  # noqa: E402
from ltx_trainer.ai_hpc_workflows.throughput import compare_strategies, throughput_metrics  # noqa: E402
from ltx_trainer.ai_hpc_workflows.tips import checklist_score, tip_detail, tips_card  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tips(_: argparse.Namespace) -> int:
    print(json.dumps(tips_card(), indent=2))
    return 0


def _cmd_tip(args: argparse.Namespace) -> int:
    detail = tip_detail(args.slug)
    if detail is None:
        print(json.dumps({"error": f"unknown tip slug: {args.slug}"}), file=sys.stderr)
        return 1
    print(json.dumps(detail, indent=2))
    return 0


def _cmd_architecture(_: argparse.Namespace) -> int:
    print(json.dumps(architecture_card(), indent=2))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    cfg = AiHpcWorkflowConfig(adaptivity=_.adaptivity)
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_phases(_: argparse.Namespace) -> int:
    print(json.dumps(default_phase_plan(), indent=2))
    return 0


def _cmd_map(args: argparse.Namespace) -> int:
    print(json.dumps(map_task_to_resource(args.task), indent=2))
    return 0


def _cmd_gravity(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            data_gravity_assessment(
                dataset_gb=args.dataset_gb,
                transfers_per_epoch=args.transfers,
                compute_hours_per_epoch=args.compute_hours,
                scratch_gb=args.scratch_gb,
            ),
            indent=2,
        )
    )
    return 0


def _cmd_orchestrator(args: argparse.Namespace) -> int:
    print(json.dumps(recommend_orchestrator(args.adaptivity), indent=2))
    return 0


def _cmd_checklist(args: argparse.Namespace) -> int:
    slugs = [s.strip() for s in args.addressed.split(",") if s.strip()]
    flags = {s: True for s in slugs}
    print(json.dumps(checklist_score(flags), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.ai_hpc_workflows.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="AI-driven HPC workflows stub CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge").set_defaults(func=_cmd_knowledge)
    sub.add_parser("paper").set_defaults(func=_cmd_paper)
    sub.add_parser("tips").set_defaults(func=_cmd_tips)
    sub.add_parser("architecture").set_defaults(func=_cmd_architecture)
    sub.add_parser("phases").set_defaults(func=_cmd_phases)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    td = sub.add_parser("tip")
    td.add_argument("slug")
    td.set_defaults(func=_cmd_tip)

    mp = sub.add_parser("map")
    mp.add_argument("task", default="train", nargs="?")
    mp.set_defaults(func=_cmd_map)

    dg = sub.add_parser("gravity")
    dg.add_argument("--dataset-gb", type=float, default=500.0)
    dg.add_argument("--transfers", type=int, default=3)
    dg.add_argument("--compute-hours", type=float, default=2.5)
    dg.add_argument("--scratch-gb", type=float, default=200.0)
    dg.set_defaults(func=_cmd_gravity)

    od = sub.add_parser("demo")
    od.add_argument("--adaptivity", default="dynamic", choices=("static", "iterative", "dynamic"))
    od.set_defaults(func=_cmd_demo)

    oc = sub.add_parser("orchestrator")
    oc.add_argument("--adaptivity", default="dynamic", choices=("static", "iterative", "dynamic"))
    oc.set_defaults(func=_cmd_orchestrator)

    cl = sub.add_parser("checklist")
    cl.add_argument(
        "--addressed",
        default="integrate_ai,data_gravity,separate_phases,containerise",
        help="comma-separated tip slugs marked as addressed",
    )
    cl.set_defaults(func=_cmd_checklist)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
