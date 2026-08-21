#!/usr/bin/env python3
"""CAST distribution-valued time series CLI (Lu et al., arXiv:2605.16919)."""

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

from ltx_trainer.cast.config import CASTConfig, PAPER_TITLE, PAPER_URL  # noqa: E402
from ltx_trainer.cast.model import CAST  # noqa: E402
from ltx_trainer.cast.pipeline import (  # noqa: E402
    aliasing_js_lower_bound,
    count_parameters,
    paper_report,
    persistence_baseline,
    train_step,
)
from ltx_trainer.cast.simplex import kl_divergence  # noqa: E402
from ltx_trainer.cast.synthetic import (  # noqa: E402
    aliasing_setup,
    compositional_sequence,
    dataset_summary,
    queue_occupancy_sequence,
)


def _cmd_knowledge(_: argparse.Namespace) -> int:
    ds = dataset_summary()
    print(
        json.dumps(
            {
                "name": "CAST",
                "paper": PAPER_TITLE,
                "url": PAPER_URL,
                "pipeline": [
                    "causal context encoder",
                    "empirical successor retrieval",
                    "persistence anchor",
                    "bounded local transport",
                ],
                "benchmark_sections": ds["sections"],
            },
            indent=2,
        )
    )
    return 0


def _cmd_tables(_: argparse.Namespace) -> int:
    print(json.dumps(paper_report(), indent=2))
    return 0


def _cmd_dataset(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_summary(), indent=2))
    return 0


def _cmd_aliasing(args: argparse.Namespace) -> int:
    setup = aliasing_setup(dim=args.dim)
    js = aliasing_js_lower_bound(setup["u_up"], setup["u_down"])
    kl_mix = float(kl_divergence(setup["mixture"].unsqueeze(0), setup["u_up"].unsqueeze(0)).item())
    print(
        json.dumps(
            {
                "js_lower_bound": js,
                "kl_mixture_vs_up": kl_mix,
                "paper_fixed_summary_kl": 0.044893,
                "dim": args.dim,
            },
            indent=2,
        )
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = CASTConfig(support_dim=args.dim, ordered_support=not args.unordered)
    model = CAST(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        if args.unordered:
            seq = compositional_sequence(batch=args.batch, length=args.length, dim=args.dim, seed=step)
        else:
            seq = queue_occupancy_sequence(batch=args.batch, length=args.length, dim=args.dim, seed=step)
        seq = seq.to(device)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 5) == 0:
            print(f"step {step}/{args.steps} loss={stats['loss']:.4f} kl={stats['kl_mean']:.4f}")

    out = {"steps": args.steps, "params": count_parameters(model), "final": log[-1] if log else {}}
    print(json.dumps(out, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = CASTConfig(support_dim=args.dim)
    model = CAST(cfg).to(device)
    model.eval()
    seq = queue_occupancy_sequence(batch=1, length=args.length, dim=args.dim, seed=args.seed).to(device)
    with torch.no_grad():
        out = model(seq)
        rolled = model.rollout(seq[:, : args.context], args.horizon)
    result = {
        "input_shape": list(seq.shape),
        "pred_shape": list(out["pred_seq"].shape),
        "rollout_shape": list(rolled.shape),
        "lambda_t_last": out["lambda_t"][0, -1].item(),
        "rho_t_last": out["rho_t"][0, -1].item(),
    }
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CAST CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")

    pa = sub.add_parser("aliasing")
    pa.add_argument("--dim", type=int, default=48)

    pt = sub.add_parser("train")
    pt.add_argument("--steps", type=int, default=20)
    pt.add_argument("--batch", type=int, default=4)
    pt.add_argument("--length", type=int, default=16)
    pt.add_argument("--dim", type=int, default=32)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--device", default="cpu")
    pt.add_argument("--unordered", action="store_true")

    pi = sub.add_parser("infer")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--length", type=int, default=24)
    pi.add_argument("--context", type=int, default=12)
    pi.add_argument("--horizon", type=int, default=4)
    pi.add_argument("--dim", type=int, default=32)
    pi.add_argument("--device", default="cpu")

    args = p.parse_args()
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "aliasing": _cmd_aliasing,
        "train": _cmd_train,
        "infer": _cmd_infer,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
