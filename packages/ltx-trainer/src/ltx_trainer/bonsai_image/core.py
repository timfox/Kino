import numpy as np

def bonsai_weight_stats(bits: float = 1.125) -> dict[str, float]:
    full_gb = 7.75
    return {"bits": bits, "gb": round(full_gb * 16.0 / bits / 16.0, 2)}
