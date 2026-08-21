"""BiLT-Autoencoder (Hohmann, arXiv:2605.11829)."""

from ltx_trainer.bilt.augment import apply_augmentation, spectral_shift
from ltx_trainer.bilt.config import BiLTConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.bilt.decoder import PhysicsConstrainedDecoder
from ltx_trainer.bilt.losses import latent_loss, log_scale_loss, reconstruction_loss, swap_penalty
from ltx_trainer.bilt.metrics import (
    mape,
    r2_score,
    table1_architecture,
    table2_training_phases,
    table3_performance,
    table_complexity,
)
from ltx_trainer.bilt.model import BiLTAutoencoder
from ltx_trainer.bilt.pipeline import count_parameters, curriculum_phase, paper_report, train_step
from ltx_trainer.bilt.scanner import BiLTScanner, CrossAttentionBlock, ImportanceGating
from ltx_trainer.bilt.synthetic import constituent_spectra, dataset_summary, mix_optical_properties, synthetic_batch

__all__ = [
    "BiLTAutoencoder",
    "BiLTConfig",
    "BiLTScanner",
    "CrossAttentionBlock",
    "ImportanceGating",
    "PAPER_TITLE",
    "PAPER_URL",
    "PhysicsConstrainedDecoder",
    "apply_augmentation",
    "constituent_spectra",
    "count_parameters",
    "curriculum_phase",
    "dataset_summary",
    "latent_loss",
    "log_scale_loss",
    "mape",
    "mix_optical_properties",
    "paper_report",
    "r2_score",
    "reconstruction_loss",
    "spectral_shift",
    "swap_penalty",
    "synthetic_batch",
    "table1_architecture",
    "table2_training_phases",
    "table3_performance",
    "table_complexity",
    "train_step",
]
