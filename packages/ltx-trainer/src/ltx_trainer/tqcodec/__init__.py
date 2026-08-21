"""TQCodec neural audio codec (He et al., arXiv:2603.01592)."""

from ltx_trainer.tqcodec.config import (
    BITRATES_KBPS,
    LOSS_WEIGHTS,
    PAPER_DOI,
    PAPER_TITLE,
    PAPER_URL,
    SAMPLE_RATE,
    TQCodecConfig,
)
from ltx_trainer.tqcodec.losses import log_spectral_distance, snr_db
from ltx_trainer.tqcodec.metrics import table3_ablation, table4_subband_baseline
from ltx_trainer.tqcodec.model import TQCodec, count_parameters, train_step
from ltx_trainer.tqcodec.pipeline import knowledge, paper_report
from ltx_trainer.tqcodec.synthetic import dataset_summary, synthetic_waveform

__all__ = [
    "BITRATES_KBPS",
    "LOSS_WEIGHTS",
    "PAPER_DOI",
    "PAPER_TITLE",
    "PAPER_URL",
    "SAMPLE_RATE",
    "TQCodec",
    "TQCodecConfig",
    "count_parameters",
    "dataset_summary",
    "knowledge",
    "log_spectral_distance",
    "paper_report",
    "snr_db",
    "synthetic_waveform",
    "table3_ablation",
    "table4_subband_baseline",
    "train_step",
]
