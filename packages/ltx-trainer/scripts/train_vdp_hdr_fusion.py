#!/usr/bin/env python3
"""Train Fusion UNet for VDP-HDR (Talegaonkar et al. arXiv:2605.11628 stage 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ltx_trainer.vdp_hdr.bracket import BracketConfig
from ltx_trainer.vdp_hdr.fusion import FusionUNet, bracket_to_linear, fuse_bracket
from ltx_trainer.vdp_hdr.losses import FusionLoss
from ltx_trainer.vdp_hdr.synthetic import synthesize_hdr_pair


class _SyntheticSet(Dataset):
    def __init__(self, size: int, side: int, cfg: BracketConfig) -> None:
        self.size = size
        self.side = side
        self.cfg = cfg

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(idx)
        torch.manual_seed(int(torch.randint(0, 2**31, (1,), generator=g).item()))
        hdr, _ldr, bracket = synthesize_hdr_pair(self.side, self.side, cfg=self.cfg)
        return bracket, hdr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train VDP-HDR Fusion UNet on synthetic HDR brackets")
    p.add_argument("-o", "--output", type=Path, required=True, help="Checkpoint path (.pt)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--side", type=int, default=128)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    cfg = BracketConfig(num_frames=args.num_frames)
    ds = _SyntheticSet(max(args.steps * 2, 64), args.side, cfg)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    fusion = FusionUNet(args.num_frames).to(args.device)
    loss_fn = FusionLoss()
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr)

    step = 0
    fusion.train()
    while step < args.steps:
        for bracket_ncfhw, hdr_chw in loader:
            if step >= args.steps:
                break
            bracket = bracket_ncfhw.to(args.device)
            hdr = hdr_chw.to(args.device)
            lin = bracket_to_linear(bracket, gamma=cfg.gamma)
            w = fusion(lin)
            pred = fuse_bracket(lin, w)
            loss, stats = loss_fn(pred, hdr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0 or step == args.steps:
                print(f"step {step}/{args.steps} fusion_mse={stats['fusion_mse']:.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "num_frames": args.num_frames,
                "gamma": cfg.gamma,
                "use_fusion_unet": True,
                "ev_span": cfg.ev_span,
                "fusion_backend": "unet",
            },
            "state_dict": fusion.state_dict(),
        },
        args.output,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
