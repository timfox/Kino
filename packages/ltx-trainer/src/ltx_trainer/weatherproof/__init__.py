"""WeatherProof UG2+ semi-supervised segmentation (Chai et al. arXiv:2605.22216)."""

from ltx_trainer.weatherproof.augment import strong_augment, weak_augment
from ltx_trainer.weatherproof.classes import IGNORE_LABEL, NUM_CLASSES, WEATHERPROOF_CLASSES
from ltx_trainer.weatherproof.losses import SemiSupervisedLoss, SemiSupervisedLossConfig
from ltx_trainer.weatherproof.metrics import mean_dice, mean_iou
from ltx_trainer.weatherproof.model import UniMatchV2Seg, UniMatchV2Config
from ltx_trainer.weatherproof.pipeline import load_weatherproof_checkpoint, predict_segmentation, train_step
from ltx_trainer.weatherproof.tta import predict_with_tta

__all__ = [
    "IGNORE_LABEL",
    "NUM_CLASSES",
    "SemiSupervisedLoss",
    "SemiSupervisedLossConfig",
    "UniMatchV2Config",
    "UniMatchV2Seg",
    "WEATHERPROOF_CLASSES",
    "load_weatherproof_checkpoint",
    "mean_dice",
    "mean_iou",
    "predict_segmentation",
    "predict_with_tta",
    "strong_augment",
    "train_step",
    "weak_augment",
]
