#!/usr/bin/env python3
"""Train ExpoCM with EACT consistency loss + optional ELC finetune (arXiv:2605.02464)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402

from ltx_trainer.expo_cm.elc_loss import ELCLoss  # noqa: E402
from ltx_trainer.expo_cm.losses import ConsistencyLoss  # noqa: E402
from ltx_trainer.expo_cm.model import ExpoCM, ExpoCMConfig  # noqa: E402
from ltx_trainer.expo_cm.synthetic import synthesize_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train ExpoCM (synthetic)")
    p.add_argument("-o", "--output-dir", default="expocm_train")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--stage", choices=("ct", "elc", "both"), default="both")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = ExpoCMConfig()
    model = ExpoCM(cfg).to(device)
    ema = copy.deepcopy(model).eval()
    for p_ema in ema.parameters():
        p_ema.requires_grad = False
    ct_loss = ConsistencyLoss()
    elc_loss = ELCLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    def ema_update(decay: float = 0.995) -> None:
        with torch.no_grad():
            for p_online, p_tgt in zip(model.parameters(), ema.parameters(), strict=True):
                p_tgt.data.mul_(decay).add_(p_online.data, alpha=1.0 - decay)

    ct_steps = args.steps if args.stage in ("ct", "both") else 0
    elc_steps = args.steps if args.stage == "elc" else (args.steps // 2 if args.stage == "both" else 0)
    total = ct_steps + elc_steps
    step = 0

    for _ in range(ct_steps):
        opt.zero_grad(set_to_none=True)
        hdr, ldr = synthesize_pair(args.size, args.size)
        hdr, ldr = hdr.to(device), ldr.to(device)
        loss, stats = ct_loss(model.net, ema.net, hdr, ldr)
        loss.backward()
        opt.step()
        ema_update()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, total // 10) == 0:
            print(f"step {step}/{total} CT={stats['loss_ct']:.6f}")
        step += 1

    for _ in range(elc_steps):
        opt.zero_grad(set_to_none=True)
        hdr, ldr = synthesize_pair(args.size, args.size)
        hdr, ldr = hdr.to(device), ldr.to(device)
        pred = model.one_step(ldr)
        loss, stats = elc_loss(pred, hdr, ldr)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, total // 10) == 0:
            print(f"step {step}/{total} ELC={stats['loss_elc']:.6f}")
        step += 1

    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, out / "expocm_checkpoint.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")
    print(f"Saved {out / 'expocm_checkpoint.pt'}")


if __name__ == "__main__":
    main()
