"""MTPano multi-task panoramic scene understanding (arXiv:2602.05330)."""

from ltx_trainer.mtpano.auxiliary import build_auxiliary_targets, edge_distance_field, image_gradient_map
from ltx_trainer.mtpano.benchmarks import benchmarks_bundle, table1_ours, table4_ours
from ltx_trainer.mtpano.config import CODE_URL, MTPanoConfig, PAPER_ARXIV, TRAIN_PANORAMAS
from ltx_trainer.mtpano.datasets import training_dataset_card
from ltx_trainer.mtpano.pd_bridgenet import PDBridgeNetStub
from ltx_trainer.mtpano.paper import evaluation_demo, framework_card
from ltx_trainer.mtpano.pipeline import evaluation_demo_run, train_step

__all__ = [
    "CODE_URL",
    "MTPanoConfig",
    "PAPER_ARXIV",
    "PDBridgeNetStub",
    "TRAIN_PANORAMAS",
    "benchmarks_bundle",
    "build_auxiliary_targets",
    "edge_distance_field",
    "evaluation_demo",
    "evaluation_demo_run",
    "framework_card",
    "image_gradient_map",
    "table1_ours",
    "table4_ours",
    "train_step",
    "training_dataset_card",
]
