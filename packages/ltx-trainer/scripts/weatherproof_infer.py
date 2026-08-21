#!/usr/bin/env python3
"""Infer semantic segmentation on WeatherProof-style images (arXiv:2605.22216)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.functional import to_tensor  # noqa: E402

from ltx_trainer.weatherproof.model import UniMatchV2Seg  # noqa: E402
from ltx_trainer.weatherproof.pipeline import load_weatherproof_checkpoint, predict_segmentation  # noqa: E402
from ltx_trainer.weatherproof.tta import predict_with_tta  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="WeatherProof segmentation infer")
    p.add_argument("image")
    p.add_argument("-o", "--output", default="seg_mask.png")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--size", type=int, default=518)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_weatherproof_checkpoint(args.checkpoint, device=device)
    else:
        model = UniMatchV2Seg().to(device)

    with Image.open(args.image) as im:
        im = im.convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR)
        img = to_tensor(im).to(device)

    if args.tta:
        pred = predict_with_tta(model, img)
    else:
        pred = predict_segmentation(model, img)

    vis = (pred.cpu().numpy().astype("uint8") * 25)
    Image.fromarray(vis).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
