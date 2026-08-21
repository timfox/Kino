#!/usr/bin/env python3
"""Train LuckyHDR on synthetic SI-HDR bursts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ltx_trainer.lucky_hdr.losses import LuckyHdrLoss
from ltx_trainer.lucky_hdr.model import LuckyHdr, LuckyHdrConfig
from ltx_trainer.lucky_hdr.synthetic import synthesize_bracket_burst


def main() -> None:
    ap = argparse.ArgumentParser(description="LuckyHDR synthetic training loop")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Checkpoint path (.pt)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    model = LuckyHdr(LuckyHdrConfig()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = LuckyHdrLoss()

    for step in range(args.steps):
        hdr = torch.rand(3, args.size, args.size, device=device)
        stack, gt, evs = synthesize_bracket_burst(hdr, num_frames=3, shake_px=1.0, seed=step)
        no_shift, _, _ = synthesize_bracket_burst(hdr, num_frames=3, shake_px=0.0, seed=step + 10_000)
        pred, warp_terms, shifts = model.forward_train(stack, no_shift, evs)
        loss, stats = loss_fn(pred, gt, warp_terms=warp_terms, shifts=shifts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
            print(f"step {step + 1}/{args.steps} loss={stats['loss_total']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.__dict__}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
