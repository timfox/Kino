#!/usr/bin/env python3
# ruff: noqa: T201
"""Export structure-from-motion data from a video file.

Usage:
    python scripts/export_sfm_from_video.py output.mp4 --output-dir ./sfm_out
    SFM_BACKEND=opencv python scripts/export_sfm_from_video.py gen.mp4 -o ./sfm_out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ltx_trainer.sfm_export import SfmBackend, colmap_available, export_sfm_from_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract camera poses and sparse 3D points from a video (SfM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Input video (.mp4, etc.)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory for sfm_summary.json, cameras.json, trajectories.npy, sparse.ply, frames/",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "colmap", "opencv"],
        default="auto",
        help="Reconstruction backend (auto prefers COLMAP when installed)",
    )
    parser.add_argument("--max-frames", type=int, default=120, help="Max frames to extract")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=2.0,
        help="Approximate frame sampling rate across the clip",
    )
    parser.add_argument("--fov-deg", type=float, default=55.0, help="Assumed horizontal FOV for OpenCV fallback")
    args = parser.parse_args()

    backend: SfmBackend = args.backend  # type: ignore[assignment]
    if backend == "auto":
        print(f"COLMAP on PATH: {colmap_available()}")

    result = export_sfm_from_video(
        args.video,
        args.output_dir,
        backend=backend,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        fov_deg=args.fov_deg,
    )
    out = Path(args.output_dir)
    print("=" * 60)
    print("SfM export complete")
    print("=" * 60)
    print(f"Backend:     {result.backend}")
    print(f"Frames:      {result.num_frames}")
    print(f"Cameras:     {len(result.cameras)}")
    print(f"3D points:   {result.num_points}")
    print(f"Summary:     {out / 'sfm_summary.json'}")
    print(f"Cameras:     {out / 'cameras.json'}")
    print(f"Trajectories:{out / 'trajectories.npy'}")
    if (out / "sparse.ply").is_file():
        print(f"Point cloud: {out / 'sparse.ply'}")
    print(f"Frames:      {out / 'frames'}/")


if __name__ == "__main__":
    main()
