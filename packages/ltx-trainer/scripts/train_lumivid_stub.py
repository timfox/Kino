#!/usr/bin/env python3
"""Train LumiVid stub on synthetic HDR/SDR pairs (arXiv:2604.11788)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

TRAINER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from ltx_trainer.lumivid.model import LumiVid  # noqa: E402
from ltx_trainer.lumivid.synthetic import synthesize_hdr_scene, synthesize_sdr_from_hdr  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train LumiVid stub")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LumiVid().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        hdr = synthesize_hdr_scene(64, 64, device=device)
        sdr = synthesize_sdr_from_hdr(hdr)
        loss, stats = model.training_step(sdr, hdr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps:
            print(f"step {step} loss={stats['loss_total']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
