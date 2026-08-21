#!/usr/bin/env python3
"""Convert SDR video to HDR preview (Tedla et al. MEVM proxy + VMM / Debevec)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ltx_trainer.sdr2hdr.pipeline import Sdr2HdrConfig, sdr_video_to_hdr, tone_map_hdr_for_preview
from ltx_trainer.video_utils import read_video, save_video


def _cfhw_from_read(video_fchw: torch.Tensor) -> torch.Tensor:
    return video_fchw.permute(1, 0, 2, 3).contiguous()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SDR → HDR video (bracket + merge)")
    p.add_argument("input", type=Path, help="Input SDR video")
    p.add_argument("--output", "-o", type=Path, required=True, help="Tone-mapped HDR preview .mp4")
    p.add_argument("--merge", choices=("debevec", "vmm"), default="debevec")
    p.add_argument("--vmm-checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-temporal-smooth", action="store_true")
    p.add_argument("--evs", default="-4,0,4")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument(
        "--preview-mode",
        choices=("filmic", "reinhard"),
        default="filmic",
        help="Tone-map HDR scene-linear frames to [0,1] preview",
    )
    p.add_argument(
        "--preview-percentile",
        type=float,
        default=0.99,
        help="Robust peak percentile for preview normalization (e.g. 0.99 ignores hot pixels)",
    )
    p.add_argument(
        "--no-input-shoulder",
        action="store_true",
        help="Skip soft shoulder on γ-encoded SDR before MEVM bracketing",
    )
    args = p.parse_args(argv)

    evs = tuple(float(x.strip()) for x in args.evs.split(",") if x.strip())
    video_fchw, fps_in = read_video(args.input, max_frames=args.max_frames)
    sdr = _cfhw_from_read(video_fchw)
    cfg = Sdr2HdrConfig(
        merge=args.merge,
        device=args.device,
        evs=evs,
        vmm_path=str(args.vmm_checkpoint) if args.vmm_checkpoint else None,
        temporal_smooth=not args.no_temporal_smooth,
        input_shoulder=not args.no_input_shoulder,
    )
    hdr, _, _ = sdr_video_to_hdr(sdr, cfg)
    preview = tone_map_hdr_for_preview(
        hdr,
        mode=args.preview_mode,
        percentile=args.preview_percentile,
    )
    save_video(preview, args.output, fps=args.fps or fps_in)
    print(f"Wrote {args.output} ({preview.shape[1]} frames @ {args.fps or fps_in} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
