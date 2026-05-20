"""Shared media suffix helpers (video, stills, Canon Raw 2)."""

from __future__ import annotations

import os
from pathlib import Path

# Canon Raw version 2 (CR2) — still photos from Canon DSLRs.
CANON_RAW_2_SUFFIXES: frozenset[str] = frozenset({".cr2"})

# Related Canon raw stills (optional ingest; same decode path as CR2).
CANON_RAW_SUFFIXES: frozenset[str] = frozenset({".cr2", ".cr3", ".crw"})

COMMON_STILL_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}
)

STILL_IMAGE_SUFFIXES: frozenset[str] = COMMON_STILL_IMAGE_SUFFIXES | CANON_RAW_SUFFIXES

VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ogv", ".qt", ".mxf"}
)

# QuickTime / MOV containers: decode via ffmpeg by default (ProRes, DNxHD, edit lists).
FFMPEG_PREFERRED_VIDEO_SUFFIXES: frozenset[str] = frozenset({".mov", ".qt"})


def path_suffix_lower(path: str | Path) -> str:
    return Path(path).suffix.lower()


def is_canon_raw2_path(path: str | Path) -> bool:
    return path_suffix_lower(path) in CANON_RAW_2_SUFFIXES


def is_canon_raw_path(path: str | Path) -> bool:
    return path_suffix_lower(path) in CANON_RAW_SUFFIXES


def is_still_image_path(path: str | Path) -> bool:
    return path_suffix_lower(path) in STILL_IMAGE_SUFFIXES


def is_video_path(path: str | Path) -> bool:
    return path_suffix_lower(path) in VIDEO_SUFFIXES


def prefer_ffmpeg_video_decode(path: str | Path) -> bool:
    """True when ingest should use ffmpeg instead of PyAV (default for ``.mov`` / ``.qt``).

    Override with ``LTX_MOV_DECODE=pyav`` to force PyAV for QuickTime, or ``LTX_VIDEO_BACKEND=ffmpeg``
    for all containers.
    """
    suf = path_suffix_lower(path)
    if suf not in FFMPEG_PREFERRED_VIDEO_SUFFIXES:
        return False
    mode = os.environ.get("LTX_MOV_DECODE", "ffmpeg").strip().lower()
    if mode in ("pyav", "av"):
        return False
    if mode in ("ffmpeg", "ff", "1", "true", "yes"):
        return True
    return True  # default: ffmpeg for MOV
