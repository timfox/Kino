#!/usr/bin/env python3
"""ND-QAT lightweight neural distinguisher CLI (Xiong et al., arXiv:2603.05791)."""

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

from ltx_trainer.nd_qat.config import NDQATConfig, PAPER_DOI, PAPER_TITLE, PAPER_URL  # noqa: E402
from ltx_trainer.nd_qat.model import GohrDistinguisher, LightweightDistinguisher  # noqa: E402
from ltx_trainer.nd_qat.pipeline import count_parameters, knowledge, paper_report, train_step  # noqa: E402
from ltx_trainer.nd_qat.synthetic import dataset_summary, synthetic_batch  # noqa: E402


def _cmd_knowledge(_: argparse.Namespace) -> int:
    print(json.dumps({**knowledge(), "paper": PAPER_TITLE}, indent=2))
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(paper_report(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_summary(), indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = NDQATConfig()
    model = LightweightDistinguisher(cfg) if args.lightweight else GohrDistinguisher(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log: list[dict[str, float]] = []
    for step in range(args.steps):
        x, y = synthetic_batch(batch=args.batch, seed=step)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, x, y)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 5) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss']:.4f} acc={stats['accuracy']:.3f}")
    print(json.dumps({"steps": args.steps, "params": count_parameters(model), "final": log[-1]}, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    model = LightweightDistinguisher(NDQATConfig()) if args.lightweight else GohrDistinguisher(NDQATConfig())
    model.eval()
    x, y = synthetic_batch(batch=4, seed=args.seed)
    with torch.no_grad():
        pred = model(x)
    out = {
        "pred": pred.tolist(),
        "labels": y.tolist(),
        "threshold": 0.505,
        "doi": PAPER_DOI,
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")
    pt = sub.add_parser("train")
    pt.add_argument("--steps", type=int, default=20)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--lightweight", action="store_true")
    pi = sub.add_parser("infer")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--lightweight", action="store_true")
    args = p.parse_args()
    handlers = {"knowledge": _cmd_knowledge, "tables": _cmd_tables, "dataset": _cmd_dataset, "train": _cmd_train, "infer": _cmd_infer}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
