"""PhysHDR-GS: physically inspired HDR Gaussian splatting (Zeng et al. arXiv:2603.28020)."""

from ltx_trainer.physthdr_gs.config import EXPOSURE_TIMES, PAPER_URL, PROJECT_URL, PhysHDRConfig
from ltx_trainer.physthdr_gs.dataset import dataset_summary
from ltx_trainer.physthdr_gs.losses import PhysHDRLoss, PhysHDRLossConfig
from ltx_trainer.physthdr_gs.metrics import TABLE2_EXP3, TABLE3_EFFICIENCY, TABLE4_ABLATION, PSNR_GAIN_OVER_HDR_GS
from ltx_trainer.physthdr_gs.model import PhysHDRGS
from ltx_trainer.physthdr_gs.pipeline import ablation_configs, evaluate_psnr, paper_report, train_step
from ltx_trainer.physthdr_gs.trainer import PhysHDRTrainer

__all__ = [
    "EXPOSURE_TIMES",
    "PAPER_URL",
    "PROJECT_URL",
    "PSNR_GAIN_OVER_HDR_GS",
    "PhysHDRConfig",
    "PhysHDRGS",
    "PhysHDRLoss",
    "PhysHDRLossConfig",
    "TABLE2_EXP3",
    "TABLE3_EFFICIENCY",
    "TABLE4_ABLATION",
    "ablation_configs",
    "dataset_summary",
    "evaluate_psnr",
    "paper_report",
    "train_step",
    "PhysHDRTrainer",
]
