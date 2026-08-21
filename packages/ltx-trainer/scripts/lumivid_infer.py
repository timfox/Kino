#!/usr/bin/env python3
"""SDR video/frame → HDR via LumiVid stub (arXiv:2604.11788)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

TRAINER_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TRAINER_SRC) not in sys.path:
    sys.path.insert(0, str(TRAINER_SRC))

from ltx_trainer.lumivid.model import LumiVid  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="LumiVid SDR→HDR infer (stub)")
    p.add_argument("sdr_image", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True, help="Tone-mapped HDR preview PNG")
    p.add_argument("--ckpt", type=Path, default=None)
    args = p.parse_args()

    sdr = to_tensor(Image.open(args.sdr_image).convert("RGB"))
    model = LumiVid()
    if args.ckpt and args.ckpt.is_file():
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    with torch.no_grad():
        hdr = model.infer(sdr.unsqueeze(0)).squeeze(0)
    peak = hdr.amax().clamp(min=1e-4)
    preview = (hdr / peak).clamp(0, 1)
    out = (preview.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(out).save(args.output)
    print(f"Wrote {args.output} (linear HDR peak={float(peak):.3f})")


if __name__ == "__main__":
    main()
