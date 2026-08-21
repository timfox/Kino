"""Shared media suffix lists and path helpers for dataset ingest and LTX preprocess."""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
        ".avi",
        ".ogv",
        ".qt",
        ".mpg",
        ".mpeg",
        ".m2v",
        ".m2ts",
        ".mts",
        ".insv",
        ".lrv",
        ".3gp",
    }
)

ALL_VIDEO_SUFFIXES = VIDEO_SUFFIXES

INSTA360_VIDEO_SUFFIXES: frozenset[str] = frozenset({".insv", ".lrv"})
INSTA360_STILL_SUFFIXES: frozenset[str] = frozenset({".insp", ".dng"})

CANON_RAW_2_SUFFIXES: frozenset[str] = frozenset({".cr2", ".cr3", ".crw"})

STILL_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
)


def _suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def is_still_image_path(path: str | Path) -> bool:
    s = _suffix(path)
    return s in STILL_IMAGE_SUFFIXES or s in CANON_RAW_2_SUFFIXES or s in INSTA360_STILL_SUFFIXES


def is_video_path(path: str | Path) -> bool:
    return _suffix(path) in ALL_VIDEO_SUFFIXES


def is_insta360_video_path(path: str | Path) -> bool:
    return _suffix(path) in INSTA360_VIDEO_SUFFIXES


def is_insta360_still_path(path: str | Path) -> bool:
    return _suffix(path) in INSTA360_STILL_SUFFIXES


def is_insta360_path(path: str | Path) -> bool:
    return is_insta360_video_path(path) or is_insta360_still_path(path)


def is_canon_raw2_path(path: str | Path) -> bool:
    return _suffix(path) in CANON_RAW_2_SUFFIXES


def is_canon_raw_path(path: str | Path) -> bool:
    """Alias used by :mod:`ltx_trainer.utils` for Canon Raw decode."""
    return is_canon_raw2_path(path)


def prefer_ffmpeg_video_decode(path: str | Path) -> bool:
    """Whether to prefer ffmpeg over PyAV for this file (QuickTime / env override)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in INSTA360_VIDEO_SUFFIXES:
        return True
    if ext in {".mov", ".qt", ".mpg", ".mpeg", ".m2v"}:
        mode = os.environ.get("LTX_MOV_DECODE", "").strip().lower()
        if mode == "pyav":
            return False
        return True
    backend = os.environ.get("LTX_VIDEO_BACKEND", "pyav").strip().lower()
    return backend == "ffmpeg"
