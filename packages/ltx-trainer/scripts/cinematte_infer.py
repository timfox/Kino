#!/usr/bin/env python3
"""Infer alpha matte from input frame + captured background plate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig  # noqa: E402
from ltx_trainer.cinematte.pipeline import _load_rgb, load_cinematte_checkpoint, matte_image  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="CineMatte background matting inference")
    p.add_argument("image", help="Input frame (actor on LED / VP stage)")
    p.add_argument("background", help="Captured background plate (inner frustum)")
    p.add_argument("-o", "--output-dir", default="cinematte_out")
    p.add_argument("--checkpoint", default="", help="Trained checkpoint (.pt); omit for untrained stub demo")
    p.add_argument("--backbone", default="stub", help="stub|dinov2_vitl14")
    p.add_argument("--size", type=int, default=768, help="Long-edge resize for inference")
    p.add_argument("--device", default="cuda")
    p.add_argument("--composite-bg", default="", help="Optional new background for composite")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_cinematte_checkpoint(args.checkpoint, device=device)
    else:
        model = CineMatte(CineMatteConfig(backbone=args.backbone)).to(device)

    size = (args.size, args.size)
    img = _load_rgb(Path(args.image), torch.device(device), size)
    bg = _load_rgb(Path(args.background), torch.device(device), size)
    new_bg = _load_rgb(Path(args.composite_bg), torch.device(device), size) if args.composite_bg else None

    result = matte_image(model, img, bg, return_composite=new_bg is not None, new_background=new_bg)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_image(result.alpha.unsqueeze(0), out / "alpha.png")
    if result.composite is not None:
        save_image(result.composite.unsqueeze(0), out / "composite.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
