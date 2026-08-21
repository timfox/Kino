"""DrawVideo sketch smoke."""

from __future__ import annotations

from typing import Any


def toy_grayscale_line() -> tuple[list[float], list[float]]:
    g = [float(i % 256) for i in range(32)]
    eroded = [max(0.0, x - 1.0) for x in g]
    return g, eroded


def toy_storyboard() -> list[dict[str, float]]:
    return [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]


def evaluation_smoke() -> dict[str, Any]:
    from ltx_trainer.drawvideo.sketch import color_dodge_sketch, shot_boundary

    g, eroded = toy_grayscale_line()
    sketch = color_dodge_sketch(g, eroded)
    shots = toy_storyboard()
    return {"num_shots": len(shots), "sketch_len": len(sketch), "boundary_flag": shot_boundary(30.0)}
