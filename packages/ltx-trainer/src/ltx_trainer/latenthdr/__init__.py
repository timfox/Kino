"""LatentHDR exposure modeling (FiLM head, EV conditioning, trainer hooks)."""

from ltx_trainer.latenthdr.ev_embedding import ExposureValueEmbedding
from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead, exposure_latent_mse
from ltx_trainer.latenthdr.training import (
    compute_latenthdr_ev_loss,
    load_exposure_head_from_checkpoint,
    sample_ev_targets_from_batch,
)

__all__ = [
    "ExposureValueEmbedding",
    "FiLMResidualExposureHead",
    "compute_latenthdr_ev_loss",
    "exposure_latent_mse",
    "load_exposure_head_from_checkpoint",
    "sample_ev_targets_from_batch",
]
