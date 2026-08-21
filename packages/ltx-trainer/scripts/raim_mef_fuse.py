#!/usr/bin/env python3
"""Fuse multi-exposure brackets with RAIM MEF stub (NTIRE 2026 Track 2)."""

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

from ltx_trainer.raim_mef.fusion import weighted_fusion  # noqa: E402
from ltx_trainer.raim_mef.metrics import leaderboard_score, lpips_proxy, psnr_from_mse, ssim_proxy  # noqa: E402
from ltx_trainer.raim_mef.model import RaimMefFusion  # noqa: E402
from ltx_trainer.raim_mef.synthetic import TEST_EV_STOPS, synthesize_sequence  # noqa: E402


def _save_rgb(path: Path, rgb: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    try:
        import cv2  # noqa: PLC0415

        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image  # noqa: PLC0415

        Image.fromarray(arr).save(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RAIM MEF fusion (learned or exposure-weighted baseline)")
    p.add_argument("-o", "--output", type=Path, help="Write fused RGB image")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic bracket stack")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--checkpoint", type=Path, help="Optional RaimMefFusion weights (.pt)")
    p.add_argument("--baseline", action="store_true", help="Exposure-weighted fusion only (no CNN)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    if not args.synthetic:
        p.error("Only --synthetic fusion is implemented in this stub (pass --synthetic)")

    scene = torch.rand(3, args.size, args.size)
    stack, gt, evs = synthesize_sequence(scene, ev_stops=TEST_EV_STOPS, shake_px=1.5)
    stack = stack.to(args.device)
    gt = gt.to(args.device)

    if args.baseline:
        fused = weighted_fusion(stack, evs)
    else:
        model = RaimMefFusion().to(args.device)
        if args.checkpoint and args.checkpoint.is_file():
            state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            fused = model(stack, evs)

    mse = float((fused - gt).pow(2).mean())
    psnr = psnr_from_mse(mse)
    ssim = ssim_proxy(fused, gt)
    lp = lpips_proxy(fused, gt)
    score = leaderboard_score(psnr, ssim, lp)

    report = {
        "ev_stops": list(evs),
        "stack_shape": list(stack.shape),
        "fused_shape": list(fused.shape),
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lp,
        "leaderboard_score": score,
        "baseline": bool(args.baseline),
    }
    if args.output:
        _save_rgb(args.output.expanduser().resolve(), fused)
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
