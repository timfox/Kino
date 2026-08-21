import numpy as np

def sega_rope_scales(latent: np.ndarray, base_scale: float = 1.0) -> np.ndarray:
    """Stub: spectral-energy guided per-frequency RoPE scaling."""
    spectrum = np.abs(np.fft.fft2(latent))
    energy = spectrum / (spectrum.sum() + 1e-8)
    return base_scale * (0.5 + energy.mean(axis=0))
