#!/usr/bin/env python3
"""Single-shot HDR recovery via VDP-HDR (bracket prior + Fusion UNet)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ltx_trainer.vdp_hdr.pipeline import load_vdp_hdr_checkpoint, recover_hdr_from_ldr
from ltx_trainer.vdp_hdr.reinhard import reinhard_tone_map


def _load_ldr(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(__import__("numpy").array(img)).float().div(255.0).permute(2, 0, 1)
    return t


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VDP-HDR single-image recovery")
    p.add_argument("input", type=Path, help="Input LDR image (8-bit sRGB)")
    p.add_argument("-o", "--output", type=Path, required=True, help="Tone-mapped preview PNG")
    p.add_argument("--checkpoint", type=Path, default=None, help="Fusion UNet checkpoint (.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-linear", type=Path, default=None, help="Optional linear HDR .pt tensor")
    args = p.parse_args(argv)

    ldr = _load_ldr(args.input)
    if args.checkpoint and args.checkpoint.is_file():
        model = load_vdp_hdr_checkpoint(args.checkpoint, device=args.device)
    else:
        from ltx_trainer.vdp_hdr.model import VdpHdr

        model = VdpHdr().to(args.device)
        model.eval()

    hdr = recover_hdr_from_ldr(model, ldr.to(args.device))
    if args.save_linear is not None:
        args.save_linear.parent.mkdir(parents=True, exist_ok=True)
        torch.save(hdr.cpu(), args.save_linear)

    preview = reinhard_tone_map(hdr)
    preview_np = (preview.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(preview_np).save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
