#!/usr/bin/env python3
"""Inspect media with ffprobe / list ffmpeg capabilities (agent-friendly JSON or text)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
try:
    import _kino_bootstrap  # noqa: F401
except ImportError:
    pass

_SRC = _SCRIPTS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ltx_trainer.ffmpeg_io import (  # noqa: E402
    detect_scene_cut_times,
    duration_seconds,
    estimate_frame_count,
    ffmpeg_version,
    list_encoders,
    probe_media,
    require_ffmpeg,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, nargs="?", help="Media file to probe")
    ap.add_argument("--json", action="store_true", help="Print probe as JSON")
    ap.add_argument("--scene-threshold", type=float, default=None, help="Run scene detect; print cut times")
    ap.add_argument("--list-encoders", action="store_true", help="List ffmpeg encoder/decoder names")
    args = ap.parse_args()

    require_ffmpeg()
    if args.list_encoders:
        enc = list_encoders()
        print(json.dumps({"ffmpeg": ffmpeg_version(), "encoders": enc}, indent=2))
        return 0

    if args.path is None:
        ap.error("path required unless --list-encoders")
    pr = probe_media(args.path)
    if args.scene_threshold is not None:
        cuts = detect_scene_cut_times(args.path, args.scene_threshold)
    else:
        cuts = []

    payload = {
        "path": str(pr.path),
        "format": pr.format_name,
        "duration_sec": pr.duration_sec,
        "estimate_frame_count": estimate_frame_count(args.path),
        "video": None,
        "audio_codec": pr.audio_codec,
        "tags": pr.tags,
        "scene_cuts": cuts,
    }
    if pr.video:
        v = pr.video
        payload["video"] = {
            "codec": v.codec_name,
            "width": v.width,
            "height": v.height,
            "avg_fps": v.avg_fps,
            "pix_fmt": v.pix_fmt,
            "color_transfer": v.color_transfer,
            "color_primaries": v.color_primaries,
            "color_space": v.color_space,
            "color_range": v.color_range,
            "nb_frames": v.nb_frames,
        }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(ffmpeg_version())
        print(f"path: {payload['path']}")
        print(f"duration: {payload['duration_sec']:.3f}s  frames~{payload['estimate_frame_count']}")
        if payload["video"]:
            vv = payload["video"]
            print(
                f"video: {vv['codec']} {vv['width']}x{vv['height']} @ {vv['avg_fps']:.3f} fps  "
                f"pix_fmt={vv['pix_fmt']} trc={vv['color_transfer']}"
            )
        if cuts:
            print(f"scene cuts ({len(cuts)}): {cuts[:20]}{'...' if len(cuts) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
