"""FFD polyomino packing (Sec. 5.3, Listing 1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.tetris.tiles import Polyomino


@dataclass
class Canvas:
    height: int
    width: int
    occupied: set[tuple[int, int]] = field(default_factory=set)

    def can_place_at(self, poly: Polyomino, origin: tuple[int, int]) -> bool:
        oi, oj = origin
        for r, c in poly.positions:
            nr, nc = oi + (r - poly.bbox_origin[0]), oj + (c - poly.bbox_origin[1])
            if nr < 0 or nc < 0 or nr >= self.height or nc >= self.width:
                return False
            if (nr, nc) in self.occupied:
                return False
        return True

    def place(self, poly: Polyomino, origin: tuple[int, int]) -> None:
        oi, oj = origin
        for r, c in poly.positions:
            nr = oi + (r - poly.bbox_origin[0])
            nc = oj + (c - poly.bbox_origin[1])
            self.occupied.add((nr, nc))


def try_place(poly: Polyomino, canvas: Canvas) -> tuple[int, int] | None:
    for i in range(canvas.height - poly.height + 1):
        for j in range(canvas.width - poly.width + 1):
            if canvas.can_place_at(poly, (i, j)):
                return i, j
    return None


def ffd_pack(
    polyominoes: list[Polyomino],
    height: int,
    width: int,
) -> list[tuple[Polyomino, int, tuple[int, int]]]:
    """Pack polyominoes into fewest canvases; returns (poly, canvas_id, position)."""
    canvases: list[Canvas] = []
    placements: list[tuple[Polyomino, int, tuple[int, int]]] = []
    for poly in sorted(polyominoes, key=len, reverse=True):
        position: tuple[int, int] | None = None
        canvas_id = -1
        for cid, canvas in enumerate(canvases):
            position = try_place(poly, canvas)
            if position is not None:
                canvas_id = cid
                break
        if position is None:
            canvas_id = len(canvases)
            canvases.append(Canvas(height=height, width=width))
            position = (0, 0)
        canvases[canvas_id].place(poly, position)
        placements.append((poly, canvas_id, position))
    return placements
