#!/usr/bin/env python3
"""Merge exposure bracket PNGs with LuckyHDR (Li et al. arXiv:2604.19976)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ltx_trainer.lucky_hdr.pipeline import merge_bracket_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="LuckyHDR bracket merge")
    ap.add_argument("frames", nargs="+", type=Path, help="Bracket PNG/JPEG paths (short→long)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--size", type=int, default=None, help="Optional square resize")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    hdr = merge_bracket_paths(
        args.frames,
        device=args.device,
        size=args.size,
        checkpoint=args.checkpoint,
    )
    arr = (hdr.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.output)
    print(f"Wrote {args.output} shape={tuple(hdr.shape)}")


if __name__ == "__main__":
    main()
