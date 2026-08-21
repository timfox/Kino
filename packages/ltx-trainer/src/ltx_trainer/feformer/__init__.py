"""FEFormer volumetric segmentation (Yang et al., arXiv:2605.11434)."""

from ltx_trainer.feformer.block import FEFormerBlock
from ltx_trainer.feformer.config import FEFormerConfig, PAPER_TITLE, PAPER_URL
from ltx_trainer.feformer.dataset import dataset_summary
from ltx_trainer.feformer.fcsb import FCSB
from ltx_trainer.feformer.fdsa import FDSA
from ltx_trainer.feformer.fgmlp import FGMLP
from ltx_trainer.feformer.losses import dice_loss, segmentation_loss
from ltx_trainer.feformer.metrics import table1_datasets, table5_complexity, table6_ablation_modules
from ltx_trainer.feformer.model import FEFormer
from ltx_trainer.feformer.pipeline import ablation_configs, count_parameters, paper_report, train_step
from ltx_trainer.feformer.synthetic import synthetic_volume
from ltx_trainer.feformer.waff import WAFF

__all__ = [
    "FCSB",
    "FDSA",
    "FEFormer",
    "FEFormerBlock",
    "FEFormerConfig",
    "FGMLP",
    "PAPER_TITLE",
    "PAPER_URL",
    "WAFF",
    "ablation_configs",
    "count_parameters",
    "dataset_summary",
    "dice_loss",
    "paper_report",
    "segmentation_loss",
    "synthetic_volume",
    "table1_datasets",
    "table5_complexity",
    "table6_ablation_modules",
    "train_step",
]
