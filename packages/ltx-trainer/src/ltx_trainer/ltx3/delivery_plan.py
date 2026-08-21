"""Delivery encode plans for composed Kino/LTX3 renders."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from ltx_trainer.ltx3.render_plan import MinuteRenderPlan


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def four_k_output_path(input_mp4: str | Path, *, hdr: bool = False) -> Path:
    src = Path(input_mp4)
    suffix = "_4k_hdr10_h265.mp4" if hdr else "_4k_h265.mp4"
    return src.with_name(src.stem + suffix)


def ffmpeg_4k_delivery_command(
    *,
    input_mp4: str | Path,
    output_mp4: str | Path | None = None,
    hdr: bool = False,
    width: int = 3840,
    height: int = 2160,
    crf: int = 18,
    preset: str = "slow",
) -> str:
    """Build a high-quality 2160p H.265 delivery encode command.

    When ``hdr`` is true this assumes the source is already HDR/PQ-like (for example
    an SDR2HDR/LatentHDR delivery pass) and writes HDR10 container metadata. It does
    not hallucinate HDR from SDR by metadata alone.
    """
    inp = Path(input_mp4)
    out = Path(output_mp4) if output_mp4 else four_k_output_path(inp, hdr=hdr)
    vf = (
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    parts = [
        "ffmpeg",
        "-y",
        "-i",
        str(inp),
        "-vf",
        vf,
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p10le",
        "-tag:v",
        "hvc1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    if hdr:
        parts.extend(
            [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
            ]
        )
    parts.append(str(out))
    return _shell_join(parts)


def minute_4k_delivery_plan(
    plan: MinuteRenderPlan,
    *,
    hdr: bool = False,
    source: str = "full",
    width: int = 3840,
    height: int = 2160,
) -> dict[str, Any]:
    """Return a no-GPU 4K encode plan for a composed minute render."""
    if source not in {"full", "temporal"}:
        raise ValueError("source must be 'full' or 'temporal'")
    inp = Path(plan.full_output)
    if source == "temporal":
        inp = inp.with_name(inp.stem + "_temporal_x2.mp4")
    out = four_k_output_path(inp, hdr=hdr)
    return {
        "source": source,
        "input": str(inp),
        "output": str(out),
        "width": width,
        "height": height,
        "hdr10_metadata": hdr,
        "note": (
            "HDR metadata assumes the input has already been HDR-lifted."
            if hdr
            else "SDR 10-bit 4K mezzanine/delivery encode."
        ),
        "command": ffmpeg_4k_delivery_command(input_mp4=inp, output_mp4=out, hdr=hdr, width=width, height=height),
    }
