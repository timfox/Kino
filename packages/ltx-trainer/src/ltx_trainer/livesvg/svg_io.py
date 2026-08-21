"""Minimal SVG helpers for path counting and recolorization (no full DiffVG)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.livesvg.recolor import assign_path_palette

_SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _local_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def count_drawable_paths(svg_text: str) -> int:
    """Count ``path`` and filled ``rect``/``circle``/``ellipse`` elements."""
    root = ET.fromstring(svg_text)
    n = 0
    for el in root.iter():
        name = _local_tag(el.tag)
        if name == "path":
            n += 1
        elif name in ("rect", "circle", "ellipse", "polygon", "polyline"):
            if el.get("fill") not in (None, "none"):
                n += 1
    return n


def recolor_svg_paths(
    svg_text: str,
    palette: Tensor | None = None,
    *,
    seed: int = 0,
) -> tuple[str, Tensor]:
    """Assign distinct ``fill`` per path for optimization-friendly supervision."""
    root = ET.fromstring(svg_text)
    paths = [el for el in root.iter() if _local_tag(el.tag) == "path"]
    if not paths:
        raise ValueError("SVG contains no <path> elements")
    if palette is None:
        palette = assign_path_palette(len(paths), seed=seed)
    for el, rgb in zip(paths, palette, strict=True):
        r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        el.set("fill", f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
        if el.get("stroke") not in (None, "none"):
            el.set("stroke", el.get("fill"))
    return ET.tostring(root, encoding="unicode"), palette


def load_svg_summary(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "num_paths": count_drawable_paths(text),
        "bytes": len(text.encode("utf-8")),
    }
