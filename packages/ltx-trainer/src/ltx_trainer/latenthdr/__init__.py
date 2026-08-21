"""LatentHDR exposure modeling (FiLM head, EV conditioning, trainer hooks)."""

from ltx_trainer.latenthdr.benchmarks import benchmarks_bundle
from ltx_trainer.latenthdr.ev_embedding import ExposureValueEmbedding
from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead, exposure_latent_mse
from ltx_trainer.latenthdr.film_unet import FiLMExposureUNet
from ltx_trainer.latenthdr.model import LatentHdr, LatentHdrConfig
from ltx_trainer.latenthdr.paper import evaluation_demo, framework_card
from ltx_trainer.latenthdr.pipeline import load_latenthdr_checkpoint, recover_hdr_l2h
from ltx_trainer.latenthdr.training import (
    compute_latenthdr_ev_loss,
    load_exposure_head_from_checkpoint,
    sample_ev_targets_from_batch,
)

__all__ = [
    "ExposureValueEmbedding",
    "FiLMExposureUNet",
    "FiLMResidualExposureHead",
    "LatentHdr",
    "LatentHdrConfig",
    "benchmarks_bundle",
    "compute_latenthdr_ev_loss",
    "evaluation_demo",
    "exposure_latent_mse",
    "framework_card",
    "load_exposure_head_from_checkpoint",
    "load_latenthdr_checkpoint",
    "recover_hdr_l2h",
    "sample_ev_targets_from_batch",
]
