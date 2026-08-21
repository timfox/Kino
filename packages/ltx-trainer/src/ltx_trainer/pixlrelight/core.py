import numpy as np

def intrinsic_conditioning_pack(rgb: np.ndarray) -> dict[str, np.ndarray]:
    albedo = np.clip(rgb, 0, 1)
    shading = rgb.mean(axis=-1, keepdims=True)
    residual = rgb - albedo * shading
    return {"albedo": albedo, "shading": shading, "residual": residual}
