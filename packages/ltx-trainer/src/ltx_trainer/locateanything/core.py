import numpy as np

def parallel_box_decode(scores: np.ndarray, threshold: float = 0.5) -> list[dict[str, float]]:
    """Stub: atomic box decode from parallel scores."""
    out: list[dict[str, float]] = []
    for i, row in enumerate(scores):
        if float(row.max()) >= threshold:
            out.append({"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9, "score": float(row.max())})
    return out
