#!/usr/bin/env python3
"""Train LumiVid LoRA flow adapter on synthetic HDR video (arXiv:2604.11788)."""

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

from ltx_trainer.lumivid.losses import FlowMatchingLoss  # noqa: E402
from ltx_trainer.lumivid.model import LumiVid, LumiVidConfig  # noqa: E402
from ltx_trainer.lumivid.pipeline import save_checkpoint  # noqa: E402
from ltx_trainer.lumivid.synthetic import synthesize_hdr_video_pair  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train LumiVid (synthetic)")
    p.add_argument("-o", "--output-dir", default="lumivid_train")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--encoding", default="logc3", choices=("logc3", "pq", "hlg", "aces"))
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cfg = LumiVidConfig(encoding=args.encoding)
    model = LumiVid(cfg).to(device)
    loss_fn = FlowMatchingLoss()
    params = list(model.dit.lora_parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, float]] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        hdr, sdr = synthesize_hdr_video_pair(frames=args.frames, height=args.size, width=args.size)
        hdr, sdr = hdr.to(device), sdr.to(device)
        z_tgt, z_ref = model.build_training_pair(hdr, sdr, ev=0.0)
        loss, stats = loss_fn(model.dit, z_tgt, z_ref)
        loss.backward()
        opt.step()
        stats["step"] = float(step)
        log.append(stats)
        if step % max(1, args.steps // 10) == 0:
            print(f"step {step}/{args.steps} flow={stats['loss_flow']:.6f}")

    save_checkpoint(model, out / "lumivid.pt")
    (out / "train_log.json").write_text(json.dumps(log[-20:], indent=2))
    print(f"Saved {out / 'lumivid.pt'}")


if __name__ == "__main__":
    main()
