#!/usr/bin/env python3
"""Enhance a low-light image with optional event voxel (.npy) or synthetic events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import _kino_bootstrap  # noqa: E402, F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from ltx_trainer.eic_lie.events import events_to_sbt_voxel  # noqa: E402
from ltx_trainer.eic_lie.model import EicLie, EicLieConfig  # noqa: E402
from ltx_trainer.eic_lie.pipeline import _load_rgb, enhance_low_light, load_eic_lie_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="EIC-LIE low-light enhancement")
    p.add_argument("low_image", help="Low-light RGB input")
    p.add_argument("-o", "--output", default="enhanced.png")
    p.add_argument("--events", default="", help="Optional events .npy [N,4] x,y,t,p")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--event-bins", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_eic_lie_checkpoint(args.checkpoint, device=device)
    else:
        model = EicLie(EicLieConfig(base_channels=args.channels, event_bins=args.event_bins)).to(device)

    low = _load_rgb(Path(args.low_image), torch.device(device))
    voxel = None
    if args.events:
        arr = np.load(args.events)
        events = [(int(r[0]), int(r[1]), float(r[2]), int(r[3])) for r in arr]
        h, w = low.shape[-2:]
        voxel = events_to_sbt_voxel(events, height=h, width=w, num_bins=model.cfg.event_bins)

    out = enhance_low_light(model, low, voxel)
    save_image(out.unsqueeze(0), Path(args.output).expanduser().resolve())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
