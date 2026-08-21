#!/usr/bin/env python3
"""TQCodec neural audio codec CLI (He et al., arXiv:2603.01592)."""

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

from ltx_trainer.tqcodec.config import PAPER_TITLE, TQCodecConfig  # noqa: E402
from ltx_trainer.tqcodec.losses import log_spectral_distance, snr_db  # noqa: E402
from ltx_trainer.tqcodec.model import TQCodec, count_parameters, train_step  # noqa: E402
from ltx_trainer.tqcodec.pipeline import knowledge, paper_report  # noqa: E402
from ltx_trainer.tqcodec.synthetic import dataset_summary, synthetic_waveform  # noqa: E402


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
    cfg = TQCodecConfig(bitrate_kbps=args.bitrate, num_codebooks=args.codebooks)
    model = TQCodec(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.8, 0.9))
    log: list[dict[str, float]] = []
    for step in range(args.steps):
        x = synthetic_waveform(batch=2, seed=step)
        opt.zero_grad(set_to_none=True)
        loss, stats = train_step(model, x)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
    print(json.dumps({"params": count_parameters(model), "final": log[-1]}, indent=2))
    return 0


def _cmd_encode(args: argparse.Namespace) -> int:
    cfg = TQCodecConfig(bitrate_kbps=args.bitrate)
    model = TQCodec(cfg)
    model.eval()
    x = synthetic_waveform(batch=1, seed=args.seed)
    with torch.no_grad():
        y, meta = model(x)
    print(
        json.dumps(
            {
                "input_shape": list(x.shape),
                "output_shape": list(y.shape),
                "lsd": log_spectral_distance(y, x),
                "snr_db": snr_db(y, x),
                "num_codebook_layers": len(meta["codes"]),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=PAPER_TITLE)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("knowledge")
    sub.add_parser("tables")
    sub.add_parser("dataset")
    pt = sub.add_parser("train")
    pt.add_argument("--steps", type=int, default=10)
    pt.add_argument("--bitrate", type=int, choices=(32, 64, 128), default=64)
    pt.add_argument("--codebooks", type=int, default=5)
    pe = sub.add_parser("encode")
    pe.add_argument("--bitrate", type=int, default=64)
    pe.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    handlers = {
        "knowledge": _cmd_knowledge,
        "tables": _cmd_tables,
        "dataset": _cmd_dataset,
        "train": _cmd_train,
        "encode": _cmd_encode,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
