import numpy as np

def overlapping_scene_chunks(n_chunks: int = 3, overlap: float = 0.25) -> list[tuple[float, float]]:
    step = (1.0 - overlap) / max(n_chunks - 1, 1)
    return [(i * step, min(1.0, i * step + overlap + step)) for i in range(n_chunks)]
