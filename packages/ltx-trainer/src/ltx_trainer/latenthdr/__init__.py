"""LatentHDR-style exposure modeling (research scaffolding).

This package is **not** wired into the main ``LtxvTrainer`` flow yet. It provides:

- Fourier + MLP conditioning on scalar EV (LatentHDR Sec. 3.3, ``φ(e)``).
- A lightweight FiLM-modulated 3D residual stack predicting ``Δz`` with ``z_e = z_base + Δz``.
- A simple ``L_ev`` MSE between predicted and VAE-encoded exposure latents.

See ``scripts/train_latenthdr_exposure.py`` and ``tools/latenthdr_trainer_qt.py`` for the separate GUI / CLI entrypoints.
"""

from ltx_trainer.latenthdr.ev_embedding import EVConditionMLP
from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead
from ltx_trainer.latenthdr.losses import exposure_latent_mse

__all__ = [
    "EVConditionMLP",
    "FiLMResidualExposureHead",
    "exposure_latent_mse",
]
