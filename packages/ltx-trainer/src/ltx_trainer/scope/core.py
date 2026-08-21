import numpy as np

def scope_action_response(features: np.ndarray, in_scope_mask: np.ndarray) -> dict[str, float]:
    """Stub: separate in-scope weapon effects vs out-of-scope camera motion."""
    n = features.shape[0]
    mask_1d = np.broadcast_to(in_scope_mask.reshape(-1)[:n], (n,))
    mask = mask_1d[:, None]
    in_e = float((features * mask).sum())
    out_e = float((features * (1.0 - mask)).sum())
    return {"in_scope": in_e, "out_scope": out_e}
