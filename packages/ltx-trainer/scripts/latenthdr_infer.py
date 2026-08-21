#!/usr/bin/env python3
"""LDR → HDR via LatentHDR (arXiv:2605.11115): latent anchor + exposure head + log merge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: F401, E402

import torch  # noqa: E402
from torchvision.transforms.functional import to_tensor  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
from PIL import Image  # noqa: E402

from ltx_trainer.latenthdr.model import LatentHdr, LatentHdrConfig  # noqa: E402
from ltx_trainer.latenthdr.pipeline import load_latenthdr_checkpoint  # noqa: E402


def _load_rgb(path: Path, size: int | None) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size:
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return to_tensor(im)


def main() -> None:
    p = argparse.ArgumentParser(description="LatentHDR l2h reconstruction")
    p.add_argument("image", help="Input LDR RGB (EV≈0 anchor)")
    p.add_argument("-o", "--output", default="latenthdr_preview.png")
    p.add_argument("--hdr-linear", default="", help="Save linear HDR tensor (.pt)")
    p.add_argument("--bracket-dir", default="", help="Save decoded γ-LDR bracket frames")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--ev-min", type=float, default=-7.0)
    p.add_argument("--ev-max", type=float, default=5.0)
    p.add_argument("--ev-step", type=float, default=1.0)
    p.add_argument("--head", choices=("unet", "film_mlp"), default="unet")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_latenthdr_checkpoint(args.checkpoint, device=device)
    else:
        cfg = LatentHdrConfig(
            ev_min=args.ev_min,
            ev_max=args.ev_max,
            ev_step=args.ev_step,
            head_type=args.head,
        )
        model = LatentHdr(cfg).to(device)

    img = _load_rgb(Path(args.image), args.size).to(device)
    model.eval()
    with torch.no_grad():
        if args.bracket_dir:
            hdr, stack = model(img, return_stack=True)
            out_dir = Path(args.bracket_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(stack):
                save_image(frame.clamp(0, 1).cpu(), out_dir / f"ev_{i:02d}.png")
        else:
            hdr = model(img)

    hdr_cpu = hdr.cpu()
    if args.hdr_linear:
        torch.save(hdr_cpu, args.hdr_linear)

    peak = hdr_cpu.amax().clamp(min=1e-6)
    preview = (hdr_cpu / peak) / (1.0 + hdr_cpu / peak)
    save_image(preview, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
