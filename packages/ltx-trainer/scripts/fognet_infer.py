#!/usr/bin/env python3
"""FogNet inference on a synthetic clip or checkpoint eval."""

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

from ltx_trainer.fognet.classes import FOGACT_CLASSES  # noqa: E402
from ltx_trainer.fognet.pipeline import load_fognet_checkpoint, predict_action  # noqa: E402
from ltx_trainer.fognet.synthetic import synthesize_clip  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="FogNet foggy action recognition")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--label", type=int, default=-1, help="Synthetic class index (-1 = random)")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_fognet_checkpoint(args.checkpoint, device=device)
    else:
        from ltx_trainer.fognet.model import FogNet

        model = FogNet().to(device)

    label = None if args.label < 0 else args.label
    clean, foggy, gt = synthesize_clip(label=label, frames=args.frames, size=args.size)
    pred, logits = predict_action(model, foggy.to(device))
    out = {
        "predicted": FOGACT_CLASSES[pred],
        "predicted_index": pred,
        "ground_truth": FOGACT_CLASSES[gt],
        "ground_truth_index": gt,
        "top5_indices": logits.topk(min(5, len(FOGACT_CLASSES))).indices.tolist(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
