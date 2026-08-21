#!/usr/bin/env python3
"""Train RAIM MEF fusion CNN on synthetic dynamic brackets."""

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

from ltx_trainer.raim_mef.model import RaimMefFusion  # noqa: E402
from ltx_trainer.raim_mef.synthetic import TRAIN_EV_STOPS, synthesize_sequence  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train RAIM MEF stub on synthetic brackets")
    p.add_argument("-o", "--output", type=Path, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model = RaimMefFusion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    log: list[dict[str, float]] = []

    model.train()
    for step in range(args.steps):
        scene = torch.rand(3, args.size, args.size, device=device)
        stack, gt, evs = synthesize_sequence(scene, ev_stops=TRAIN_EV_STOPS, shake_px=2.0)
        opt.zero_grad(set_to_none=True)
        loss, stats = model.training_step(stack, gt, evs)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(
                f"step {step}/{args.steps} loss={stats['loss_l1']:.4f} "
                f"score={stats['leaderboard_score']:.2f}"
            )

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(json.dumps({"checkpoint": str(out), "steps": args.steps, "log_tail": log[-3:]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
