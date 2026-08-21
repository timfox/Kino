#!/usr/bin/env python3
"""LF-Diff bracket HDR reconstruction CLI (arXiv:2503.07351)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.lf_diff.benchmarks import PAPER_ARXIV, PAPER_TITLE, benchmarks_bundle  # noqa: E402
from ltx_trainer.lf_diff.model import LfDiff, LfDiffConfig  # noqa: E402
from ltx_trainer.lf_diff.paper import framework_card  # noqa: E402
from ltx_trainer.lf_diff.pipeline import evaluation_demo, pipeline_demo, train_step  # noqa: E402
from ltx_trainer.lf_diff.synthetic import synthesize_exposure_bracket, synthesize_hdr_scene  # noqa: E402
from ltx_trainer.lf_diff.tonemap import tonemap_l1  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "name": "LF-Diff",
                "arxiv": PAPER_ARXIV,
                "paper": PAPER_TITLE,
                "framework": framework_card(),
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(benchmarks_bundle(), indent=2))
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    from ltx_trainer.lf_diff.mock import evaluation_smoke

    print(json.dumps(evaluation_smoke(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    print(json.dumps(pipeline_demo(device=args.device), indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    print(json.dumps(evaluation_demo(device=args.device), indent=2))
    return 0


def _cmd_tonemap(args: argparse.Namespace) -> int:
    hdr = synthesize_hdr_scene(args.size, args.size)
    pred = hdr * 0.95
    print(json.dumps({"tonemap_l1": float(tonemap_l1(pred, hdr))}, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = LfDiff(LfDiffConfig()).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        hdr = synthesize_hdr_scene(args.size, args.size).to(device)
        brackets = [b.to(device) for b in synthesize_exposure_bracket(hdr)]
        stats = train_step(model, brackets, hdr, optimizer=opt)
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss']:.4f}")

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "lf_diff.pt")
    print(json.dumps({"steps": args.steps, "log_tail": log[-3:]}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [
        ("knowledge", _cmd_knowledge),
        ("tables", _cmd_tables),
        ("smoke", _cmd_smoke),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)

    sp = sub.add_parser("demo")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_demo)

    sp = sub.add_parser("eval")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("tonemap")
    sp.add_argument("--size", type=int, default=64)
    sp.set_defaults(func=_cmd_tonemap)

    sp = sub.add_parser("train")
    sp.add_argument("-o", "--output-dir", default="lf_diff_train")
    sp.add_argument("--steps", type=int, default=100)
    sp.add_argument("--size", type=int, default=64)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=_cmd_train)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
