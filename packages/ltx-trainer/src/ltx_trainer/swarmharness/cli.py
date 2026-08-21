"""CLI for SwarmHarness local swarm demos (arXiv:2605.28764)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ltx_trainer.swarmharness.agent import demo_autonomous_agent
from ltx_trainer.swarmharness.bootstrap import BootstrapConfig, bootstrap_card, parse_peer_list
from ltx_trainer.swarmharness.harness_bridge import harness_bridge_card
from ltx_trainer.swarmharness.orchestrator import demo_local_swarm
from ltx_trainer.swarmharness.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
)


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_framework(_args: argparse.Namespace) -> int:
    _print_json(framework_card())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    if args.mode == "paper":
        _print_json(evaluation_demo(seed=args.seed))
    else:
        _print_json(demo_local_swarm(seed=args.seed))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    peers = parse_peer_list(args.peers) if args.peers else []
    cfg = BootstrapConfig(
        mdns_enabled=not args.no_mdns,
        dns_seeds=[] if args.private else list(BootstrapConfig().dns_seeds),
        explicit_peers=peers,
    )
    _print_json(bootstrap_card(cfg))
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    root = Path(args.root)
    _print_json(harness_bridge_card(root))
    return 0


def cmd_knowledge(_args: argparse.Namespace) -> int:
    _print_json(knowledge_card())
    return 0


def cmd_benchmarks(_args: argparse.Namespace) -> int:
    _print_json(benchmarks_bundle())
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    _print_json(demo_autonomous_agent(seed=args.seed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SwarmHarness — decentralised HarnessAPI skill swarm utilities",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("framework", help="Print framework card (JSON)")
    sub.add_parser("knowledge", help="Positioning, security, deployment (JSON)")
    sub.add_parser("benchmarks", help="Paper tables bundle (JSON)")

    d = sub.add_parser("demo", help="Run end-to-end smoke demo")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument(
        "--mode",
        choices=("paper", "local"),
        default="local",
        help="paper=registry demo; local=signed ledger swarm",
    )

    b = sub.add_parser("bootstrap", help="Print bootstrap plan")
    b.add_argument("--peers", type=str, default="", help="Comma-separated host:port list")
    b.add_argument("--private", action="store_true", help="Private swarm (no DNS seeds)")
    b.add_argument("--no-mdns", action="store_true")

    s = sub.add_parser("skills", help="List skills under a HarnessAPI skills root")
    s.add_argument("root", type=str, help="Path to skills/ directory")

    a = sub.add_parser("agent", help="Autonomous agent goal dispatch demo")
    a.add_argument("--seed", type=int, default=0)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "framework": cmd_framework,
        "knowledge": cmd_knowledge,
        "benchmarks": cmd_benchmarks,
        "demo": cmd_demo,
        "bootstrap": cmd_bootstrap,
        "skills": cmd_skills,
        "agent": cmd_agent,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
