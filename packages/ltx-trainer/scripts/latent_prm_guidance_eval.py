#!/usr/bin/env python3
"""Latent PRM guidance CLI (Bitan et al., arXiv:2606.05518)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

from ltx_trainer.latent_prm_guidance.benchmarks import (  # noqa: E402
    paired_significance,
    summary_anchors,
    table_1_main,
    table_2_branch_selection,
    table_4_wlt,
    table_5_directions,
    table_6_ablation,
    trajectory_analysis,
)
from ltx_trainer.latent_prm_guidance.config import LatentPrmGuidanceConfig  # noqa: E402
from ltx_trainer.latent_prm_guidance.latent import alignment_transform_card  # noqa: E402
from ltx_trainer.latent_prm_guidance.mock import evaluation_smoke  # noqa: E402
from ltx_trainer.latent_prm_guidance.paper import knowledge_bundle, paper_card  # noqa: E402
from ltx_trainer.latent_prm_guidance.pipeline import run_demo  # noqa: E402
from ltx_trainer.latent_prm_guidance.prm import inference_card, prm_architecture_card, training_card
from ltx_trainer.latent_prm_guidance.paratrans import paratrans_card
from ltx_trainer.latent_prm_guidance.reward import reward_card


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps(knowledge_bundle(), indent=2))
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print(json.dumps(paper_card(), indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "table_1": table_1_main(),
                "table_2": table_2_branch_selection(),
                "table_4_wlt": table_4_wlt(),
                "table_5_directions": table_5_directions(),
                "table_6_ablation": table_6_ablation(),
                "paired_significance": paired_significance(),
                "summary": summary_anchors(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_method(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "latent_alignment": alignment_transform_card(),
                "prm_architecture": prm_architecture_card(),
                "prm_training": training_card(),
                "prm_inference": inference_card(),
                "reward": reward_card(),
                "trajectory_analysis": trajectory_analysis(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_paratrans(_: argparse.Namespace) -> int:
    print(json.dumps(paratrans_card(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    cfg = LatentPrmGuidanceConfig(
        latent_steps=args.latent_steps,
        branches_test=args.branches,
    )
    print(json.dumps(run_demo(cfg), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Latent PRM guidance for ParaTrans stub")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (
        ("knowledge", _cmd_knowledge),
        ("paper", _cmd_paper),
        ("tables", _cmd_tables),
        ("method", _cmd_method),
        ("paratrans", _cmd_paratrans),
        ("smoke", _cmd_smoke),
    ):
        sub.add_parser(name).set_defaults(func=fn)

    demo = sub.add_parser("demo")
    demo.add_argument("--latent-steps", type=int, default=12)
    demo.add_argument("--branches", type=int, default=8)
    demo.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
