import numpy as np

def physx_bench_scores(**kwargs: float) -> dict[str, float]:
    return {k: float(v) for k, v in kwargs.items()}
