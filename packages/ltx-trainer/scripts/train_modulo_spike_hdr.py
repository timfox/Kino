#!/usr/bin/env python3
"""Train ModuloSpikeHdr unwrapper on synthetic modulo pairs (arXiv:2604.14632)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

TRAINER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from ltx_trainer.modulo_spike_hdr.losses import UnwrapLoss  # noqa: E402
from ltx_trainer.modulo_spike_hdr.model import ModuloSpikeHdrUnwrapper  # noqa: E402
from ltx_trainer.modulo_spike_hdr.synthetic import synthesize_modulo_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train modulo spike HDR stub")
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModuloSpikeHdrUnwrapper().to(device)
    loss_fn = UnwrapLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        hdr = torch.rand(3, 64, 64, device=device) * 0.7 + 0.05
        modulo, _, i_mu_gt = synthesize_modulo_pair(hdr, period=256.0)
        i_mu, i_lin, _ = model(modulo)
        loss, stats = loss_fn(i_mu, i_lin, i_mu_gt, hdr, modulo * 256.0)
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
