#!/usr/bin/env python3
"""P-MLIP inference with uncertainty ensemble."""

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

from ltx_trainer.pmlip.egnn import PEGNN  # noqa: E402
from ltx_trainer.pmlip.model import PMLIPConfig  # noqa: E402
from ltx_trainer.pmlip.pipeline import evaluate_batch, load_pmlip_checkpoint, predict_with_uncertainty  # noqa: E402
from ltx_trainer.pmlip.synthetic import synthesize_nbody  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="P-MLIP uncertainty inference")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("-k", type=int, default=50, help="Ensemble size at inference")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.checkpoint:
        model = load_pmlip_checkpoint(args.checkpoint, device=device)
        pegnn = model.model
    else:
        pegnn = PEGNN().to(device)

    h, x, edge_index, target = synthesize_nbody(seed=args.seed)
    h, x, edge_index, target = h.to(device), x.to(device), edge_index.to(device), target.to(device)
    out = predict_with_uncertainty(pegnn, h, x, edge_index, k=args.k)
    metrics = evaluate_batch(pegnn, h, x, edge_index, target, k=args.k)
    result = {
        "k": args.k,
        "mean_shape": list(out["mean"].shape),
        "variance_mean": float(out["variance"].mean()),
        **metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
